import time
import re
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from .loader import load_qwen_model, load_moe_model
from .patcher import patch_model, patch_qwen
from .kv_cache import StaticKVCache


class FastServer:
    """
    High-Throughput Server for MoE (Hunyuan HYV3, Qwen2-MoE, DeepSeek) and Dense (Qwen2.5/3.5, LLaMA) models.
    Supports 4+ concurrent rollout streams with native FP8 / Unsloth / BitsAndBytes + Sword Pure FlashAttention.
    Includes zero-allocation Static KV caching and async decode pipelining.
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

        # Determine device dynamically from model parameters or explicit argument
        model_device = getattr(model, "device", None)
        if model_device is None:
            try:
                model_device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                model_device = None

        if device is not None:
            self.device = torch.device(device)
        elif model_device is not None:
            self.device = model_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration attributes (supports HYV3 MoE, Qwen, multimodal text_config)
        cfg = getattr(model, "config", None)
        t_cfg = getattr(cfg, "text_config", cfg)
        num_layers = getattr(t_cfg, "num_hidden_layers", 16)
        num_kv_heads = getattr(t_cfg, "num_key_value_heads", getattr(t_cfg, "num_attention_heads", 16))
        hidden_size = getattr(t_cfg, "hidden_size", 2048)
        num_heads = getattr(t_cfg, "num_attention_heads", 16)
        head_dim = getattr(t_cfg, "head_dim", hidden_size // num_heads)
        
        # In FP8 models, use bfloat16/float16 for KV Cache to preserve precision and Flash SDPA compatibility
        raw_dtype = getattr(model, "dtype", torch.bfloat16)
        if raw_dtype is None or "float8" in str(raw_dtype):
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        else:
            dtype = raw_dtype

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

        # Attach cache to attention layers and guarantee layer_idx
        cur_layer = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module._sword_static_cache = self.static_cache
                if not hasattr(module, "layer_idx") or module.layer_idx is None:
                    m = re.search(r"layers?\.(\d+)", name)
                    if m:
                        module.layer_idx = int(m.group(1))
                    else:
                        module.layer_idx = cur_layer
                        cur_layer += 1

        self.compile_decode = compile_decode
        self.decode_fn = self.model
        if compile_decode and self.device.type == "cuda":
            print("[Sword] Compiling decode loop with TorchInductor (dynamic=True, zero graph recapture)...")
            try:
                # Use dynamic=True to avoid CUDA Graph recompilation stalls when sequence length grows
                self.decode_fn = torch.compile(self.model, dynamic=True)
            except Exception as e:
                print(f"[Sword] torch.compile note: {e}. Defaulting to optimized eager mode.")
                self.decode_fn = self.model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "tencent/Hy-MT2-30B-A3B-FP8",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_concurrency: int = 4,
        max_seq_len: int = 2048,
        compile_decode: bool = False,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Clean one-line factory method for serving MoE or Dense models:
        - For FP8 MoE (e.g. tencent/Hy-MT2-30B-A3B-FP8): uses load_moe_model
        - For 4-bit/8-bit models: uses load_qwen_model
        """
        is_moe_or_fp8 = any(x in model_name_or_path.lower() for x in ["fp8", "moe", "hy-", "hy_", "hunyuan", "deepseek"])
        if is_moe_or_fp8 and not (load_in_4bit or load_in_8bit):
            model, tokenizer = load_moe_model(
                model_name_or_path=model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_seq_length=max_seq_len,
            )
        else:
            model, tokenizer = load_qwen_model(
                model_name_or_path=model_name_or_path,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                max_seq_length=max_seq_len,
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
        self.static_cache.set_pos(0)
        prefill_pos_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=prefill_pos_ids,
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
        self.static_cache.set_pos(curr_pos)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        active_mask = torch.ones((bsz, 1), dtype=torch.bool, device=self.device)

        # Pre-allocate decode_pos_ids buffer (zero allocation per step)
        decode_pos_ids = torch.empty((bsz, 1), dtype=torch.long, device=self.device)

        # High-Speed Async Decode Loop (zero CPU-GPU sync stalls per token)
        for step in range(1, max_new_tokens):
            self.static_cache.set_pos(curr_pos)
            decode_pos_ids.fill_(curr_pos)
            out = self.decode_fn(
                input_ids=next_token,
                position_ids=decode_pos_ids,
                sword_static_cache=self.static_cache,
                start_pos=curr_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]
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
                # Check every 8 steps to amortize GPU→CPU sync cost while
                # limiting max wasted tokens per stream to 7 (vs 15 at interval=16)
                if step % 8 == 0 and not active_mask.any():
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
        print(f"Compilation:       {'Enabled (dynamic=True)' if getattr(self, 'compile_decode', False) else 'Disabled (Optimized Eager SDPA)'}")
        print("=" * 72)

        # -----------------------------------------------------------------
        # 1. BEFORE: Standard Generation (Stock HuggingFace baseline)
        # -----------------------------------------------------------------
        print("\n[*] Running [BEFORE] baseline (standard HuggingFace generation)...")
        from .patcher import unpatch_model, patch_model
        unpatch_model(self.model)

        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
        
        # Warm up GPU kernels so initial initialization does not distort metrics
        if self.device.type == "cuda":
            _ = self.model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
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
        # 2. AFTER: Sword Speed Engine (Flash SDPA + Static KV + Fast MoE + Async Pipeline)
        # -----------------------------------------------------------------
        print("[*] Running [AFTER] with Sword Speed Engine (Flash SDPA + Static KV + Fast MoE + Pipeline)...")
        patch_model(self.model, mode="flash", patch_moe=True)
        for module in self.model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module._sword_static_cache = self.static_cache

        # Warm up Sword Speed Engine so first-call JIT kernel initialization does not distort metrics
        if self.device.type == "cuda":
            _ = self.serve(prompts, max_new_tokens=2, temperature=0.0)
            torch.cuda.synchronize()

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


# Aliases for architecture-specific imports and backward compatibility
FastMoEServer = FastServer
FastQwenServer = FastServer
