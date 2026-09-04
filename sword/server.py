import time
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from .loader import load_qwen_model
from .patcher import patch_qwen
from .kv_cache import StaticKVCache


class FastQwenServer:
    """
    High-Throughput Server for Qwen (Qwen2, Qwen2.5, Qwen3, Qwen3.5) models.
    Supports 4+ concurrent rollout streams with Unsloth / BitsAndBytes + Sword Pure FlashAttention.
    Includes CUDA graph compilation (mode='reduce-overhead') and async decode pipelining.
    """
    def __init__(
        self,
        model,
        tokenizer,
        max_concurrency: int = 4,
        max_seq_len: int = 2048,
        device: Optional[str] = None,
        compile_decode: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        self.max_concurrency = max_concurrency
        self.max_seq_len = max_seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Model configuration attributes (support both standard and multimodal text_config)
        cfg = getattr(model, "config", None)
        t_cfg = getattr(cfg, "text_config", cfg)
        num_layers = getattr(t_cfg, "num_hidden_layers", 16)
        num_kv_heads = getattr(t_cfg, "num_key_value_heads", getattr(t_cfg, "num_attention_heads", 16))
        hidden_size = getattr(t_cfg, "hidden_size", 4096)
        num_heads = getattr(t_cfg, "num_attention_heads", 16)
        head_dim = getattr(t_cfg, "head_dim", hidden_size // num_heads)
        dtype = getattr(model, "dtype", torch.bfloat16)

        # Allocate Static KV Cache for zero-allocation decode
        self.static_cache = StaticKVCache(
            num_layers=num_layers,
            max_batch_size=max_concurrency,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )

        # Attach cache to attention layers
        for module in self.model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module._sword_static_cache = self.static_cache

        self.compile_decode = compile_decode
        self.decode_fn = self.model
        if compile_decode and self.device.type == "cuda":
            print("[Sword] Compiling decode loop with mode='reduce-overhead' (CUDA Graphs enabled)...")
            try:
                # Wrap forward pass in torch.compile with CUDA graph reduction
                self.decode_fn = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[Sword] torch.compile note: {e}. Defaulting to optimized eager mode.")
                self.decode_fn = self.model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        max_concurrency: int = 4,
        max_seq_len: int = 2048,
        compile_decode: bool = False,
    ):
        """Clean one-line factory method for Colab / Modal cells."""
        model, tokenizer = load_qwen_model(
            model_name_or_path=model_name_or_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_concurrency=max_concurrency,
            max_seq_len=max_seq_len,
            compile_decode=compile_decode,
        )

    @torch.inference_mode()
    def serve(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Serves concurrent prompt requests with async pipeline and zero sync stalls.
        """
        bsz = len(prompts)
        assert bsz <= self.max_concurrency, f"Prompt batch ({bsz}) exceeds max_concurrency ({self.max_concurrency})"

        # Tokenize with left-padding for generation
        enc = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        prompt_len = input_ids.shape[1]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Prefill phase
        self.static_cache.reset(list(range(bsz)))
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sword_static_cache=self.static_cache,
            start_pos=0,
            use_cache=True,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        past_key_values = getattr(outputs, "past_key_values", None)
        next_token_logits = logits[:, -1, :]

        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [next_token]
        curr_pos = prompt_len
        tokens_per_stream = [1] * bsz
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        active_mask = torch.ones((bsz, 1), dtype=torch.bool, device=self.device)

        # High-Speed Async Decode Loop (zero CPU-GPU sync stalls per token)
        for step in range(1, max_new_tokens):
            out = self.decode_fn(
                input_ids=next_token,
                sword_static_cache=self.static_cache,
                start_pos=curr_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_logits = out.logits[:, 0, :] if hasattr(out, "logits") else out[0][:, 0, :]
            past_key_values = getattr(out, "past_key_values", past_key_values)

            if temperature > 0.0:
                probs = F.softmax(step_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            generated.append(next_token)
            curr_pos += 1

            if eos_id is not None:
                active_mask = active_mask & (next_token != eos_id)
                # Check for all streams finished every 16 tokens to avoid interrupting GPU pipeline
                if step % 16 == 0 and not active_mask.any():
                    break

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

        all_tokens = torch.cat([input_ids] + generated, dim=1)
        responses = self.tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
        total_generated_tokens = (len(generated)) * bsz
        tps = total_generated_tokens / total_time if total_time > 0 else 0.0
        stream_speed = tps / bsz

        return {
            "responses": responses,
            "latency_s": total_time,
            "total_tokens": total_generated_tokens,
            "tokens_per_stream": [len(generated)] * bsz,
            "stream_tps": [stream_speed] * bsz,
            "total_tps": tps,
        }

    @torch.inference_mode()
    def benchmark_before_after(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """
        Directly compares standard HuggingFace/Unsloth generation (BEFORE)
        vs Sword Pure FlashAttention + StaticKVCache (AFTER) on concurrent streams.
        Prints and returns detailed speed metrics for each stream.
        """
        if prompts is None:
            prompts = [
                "Explain the architecture of NVIDIA Blackwell GPUs in one sentence.",
                "How does Gated DeltaNet improve attention efficiency?",
                "Write a fast Python function to calculate matrix multiplication.",
                "Why is static KV caching faster than dynamic concatenation?",
            ]

        bsz = len(prompts)
        print("=" * 72)
        print(f" BENCHMARK: {bsz}-CONCURRENCY SERVING (BEFORE vs AFTER)")
        print("=" * 72)
        print(f"Concurrency:       {bsz} concurrent streams")
        print(f"Target GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Local Dev)'}")
        print(f"Tokens/stream:     {max_new_tokens} new tokens")
        print(f"CUDA Graphs:       {'Enabled' if getattr(self, 'compile_decode', False) else 'Disabled'}")
        print("=" * 72)

        # -----------------------------------------------------------------
        # 1. BEFORE: Standard Generation (Dynamic allocation / standard cache)
        # -----------------------------------------------------------------
        print("\n[*] Running [BEFORE] baseline (standard generation)...")
        # Temporarily detach static cache to measure baseline
        for module in self.model.modules():
            if hasattr(module, "_sword_static_cache"):
                module._sword_static_cache = None

        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        baseline_out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        before_time = time.perf_counter() - t0
        before_tokens = (baseline_out.shape[1] - enc["input_ids"].shape[1]) * bsz
        before_tps = before_tokens / before_time if before_time > 0 else 0.0
        before_stream_tps = [before_tps / bsz] * bsz

        # -----------------------------------------------------------------
        # 2. AFTER: Sword Speed Engine (Flash SDPA + Static KV + Async Pipeline)
        # -----------------------------------------------------------------
        print("[*] Running [AFTER] with Sword Speed Engine (Flash SDPA + Static KV + Pipeline)...")
        for module in self.model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module._sword_static_cache = self.static_cache

        after_results = self.serve(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
        after_time = after_results["latency_s"]
        after_tokens = after_results["total_tokens"]
        after_tps = after_results["total_tps"]
        after_stream_tps = after_results["stream_tps"]

        # -----------------------------------------------------------------
        # Output Speed Comparison Table
        # -----------------------------------------------------------------
        print("\n" + "=" * 72)
        print(f"{'Stream':<10}{'BEFORE (TPS)':<18}{'AFTER (TPS)':<18}{'Speedup':<12}")
        print("-" * 72)
        for i in range(bsz):
            sp = after_stream_tps[i] / before_stream_tps[i] if before_stream_tps[i] > 0 else 1.0
            print(f"Stream {i+1:<3}{before_stream_tps[i]:<18.2f}{after_stream_tps[i]:<18.2f}{sp:<10.2f}x")
        print("-" * 72)
        total_speedup = after_tps / before_tps if before_tps > 0 else 1.0
        print(f"{'TOTAL':<10}{before_tps:<18.2f}{after_tps:<18.2f}{total_speedup:<10.2f}x")
        print("=" * 72)
        print(f"Target of 20+ TPS for {bsz} concurrency: {'ACHIEVED' if after_tps >= 20.0 else 'CHECK RUN'}\n")

        return {
            "before_time": before_time,
            "before_total_tps": before_tps,
            "before_stream_tps": before_stream_tps,
            "after_time": after_time,
            "after_total_tps": after_tps,
            "after_stream_tps": after_stream_tps,
            "speedup": total_speedup,
        }
