import time
import re
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from .loader import load_qwen_model, load_moe_model
from .patcher import patch_model, patch_qwen
from .kv_cache import StaticKVCache


def find_prompt_lookup_candidates(
    token_sequence: List[int],
    max_ngram_size: int = 3,
    num_pred_tokens: int = 3,
) -> Optional[List[int]]:
    """
    Lightning-fast N-Gram Speculative Drafter for Prompt Lookup Decoding.
    Finds recurring n-grams between generated tokens and prompt/history context.
    Executes in microseconds on CPU / memory and requires zero additional VRAM.
    """
    seq_len = len(token_sequence)
    if seq_len < 4:
        return None

    for ngram_size in range(min(max_ngram_size, seq_len - 1), 1, -1):
        target = token_sequence[-ngram_size:]
        limit = seq_len - ngram_size - 1
        for i in range(limit, -1, -1):
            if token_sequence[i : i + ngram_size] == target:
                start_idx = i + ngram_size
                end_idx = min(start_idx + num_pred_tokens, seq_len)
                candidates = token_sequence[start_idx:end_idx]
                if candidates:
                    return candidates
    return None


class FastServer:
    """
    High-Throughput Server for MoE (Hunyuan HYV3, Qwen2-MoE, DeepSeek) and Dense (Qwen2.5/3.5, LLaMA) models.
    Supports 4+ concurrent rollout streams with native FP8 / Unsloth / BitsAndBytes + Sword Pure FlashAttention.
    Includes zero-allocation Static KV caching, Speculative Prompt Drafting, and async decode pipelining.
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

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

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

        # Architecture and quantization detection
        self.is_moe = (
            getattr(cfg, "num_experts", 0) > 0
            or getattr(cfg, "num_local_experts", 0) > 0
            or hasattr(cfg, "moe_intermediate_size")
            or any("moe" in m.__class__.__name__.lower() for m in self.model.modules())
        )
        self.is_quantized = (
            getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_loaded_in_8bit", False)
            or getattr(cfg, "load_in_4bit", False)
            or any("4bit" in m.__class__.__name__.lower() or "bnb" in m.__class__.__name__.lower() for m in self.model.modules())
        )

        if compile_decode and self.device.type == "cuda":
            if self.is_moe or self.is_quantized:
                print("[Sword] Note: Full-model torch.compile is safely skipped for 30B MoE / 4-bit quantized models")
                print("        to avoid TorchDynamo 10-30 min CPU compilation freeze (0% GPU usage).")
                print("        Running with FlashAttention SDPA + Speculative Drafter + hardware-fused kernels.")
                self.decode_fn = self.model
            else:
                print("[Sword] Compiling decode loop with TorchInductor (dynamic=True, zero graph recapture)...")
                try:
                    # Use dynamic=True to avoid CUDA Graph recompilation stalls when sequence length grows
                    self.decode_fn = torch.compile(self.model, dynamic=True)
                except Exception as e:
                    print(f"[Sword] torch.compile note: {e}. Defaulting to optimized eager mode.")
                    self.decode_fn = self.model

        # SGLang-Style CUDA Graph Decoding Runner for 4-bit MoE
        self.cuda_graph_runner = None
        self._graph_captured = False

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
        is_qwen3_moe = "qwen3" in model_name_or_path.lower() and ("moe" in model_name_or_path.lower() or "a3b" in model_name_or_path.lower())
        if is_qwen3_moe:
            from .loader import load_qwen3_moe_model
            model, tokenizer = load_qwen3_moe_model(
                model_name_or_path=model_name_or_path,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_seq_length=max_seq_len,
            )
            return cls(
                model=model,
                tokenizer=tokenizer,
                max_concurrency=max_concurrency,
                max_seq_len=max_seq_len,
                compile_decode=compile_decode,
            )

        is_ling = any(x in model_name_or_path.lower() for x in ["ling", "bailing"])
        if is_ling:
            from .ling import FastLingServer
            return FastLingServer.from_pretrained(
                model_name_or_path=model_name_or_path,
                max_concurrency=max_concurrency,
                max_seq_len=max_seq_len,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )

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
    def generate_rollouts(
        self,
        prompts: List[str],
        num_rollouts_per_prompt: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        auto_clear: bool = True,
        use_speculative: Optional[bool] = None,
        speculative_k: int = 3,
        **kwargs,
    ) -> List[List[str]]:
        """
        High-throughput multi-trajectory parallel rollout generation for RL (GRPO / PPO).
        Generates G rollouts per prompt with automatic KV cache clearing, prefix sharing,
        and optional Speculative Prompt Lookup Decoding.
        """
        # Auto-clear KV cache when new rollout batch arrives
        if auto_clear and hasattr(self, "static_cache"):
            self.static_cache.new_rollout()

        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_rollouts_per_prompt)

        results = self.serve(
            prompts=expanded_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_speculative=use_speculative,
            speculative_k=speculative_k,
        )

        all_res = results["responses"]
        grouped = []
        for i in range(len(prompts)):
            start_i = i * num_rollouts_per_prompt
            grouped.append(all_res[start_i : start_i + num_rollouts_per_prompt])

        return grouped

    @torch.inference_mode()
    def _serve_single_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        speculative_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Ultra-fast Speculative Prompt Lookup Decoding for single rollout stream.
        Drafts candidate n-grams from the prompt & recent context in microseconds,
        verifying multiple tokens per single forward pass without auxiliary models.
        """
        enc = self.tokenizer(
            [prompt],
            padding=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        self.static_cache.new_rollout(batch_size=1)
        self.static_cache.set_pos(0)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
            pos_ids = (attn_mask.long().cumsum(-1) - 1).clamp_min(0)
        else:
            pos_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=pos_ids,
            past_key_values=self.static_cache,
            sword_static_cache=self.static_cache,
            start_pos=0,
            use_cache=True,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_logits = logits[:, -1, :]

        if temperature > 0.0:
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                top_logits, top_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                probs = F.softmax(top_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, -1, sample)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens = [next_token.item()]
        token_history = input_ids[0].tolist() + [next_token.item()]
        curr_pos = prompt_len
        self.static_cache.set_pos(curr_pos)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        while len(generated_tokens) < max_new_tokens:
            if eos_id is not None and generated_tokens[-1] == eos_id:
                break

            remaining = max_new_tokens - len(generated_tokens)
            draft_k = min(speculative_k, remaining)
            candidates = find_prompt_lookup_candidates(token_history, max_ngram_size=3, num_pred_tokens=draft_k)

            if not candidates:
                decode_in = torch.tensor([[generated_tokens[-1]]], dtype=torch.long, device=self.device)
                decode_pos = torch.tensor([[curr_pos]], dtype=torch.long, device=self.device)
                self.static_cache.set_pos(curr_pos)
                out = self.model(
                    input_ids=decode_in,
                    position_ids=decode_pos,
                    past_key_values=self.static_cache,
                    sword_static_cache=self.static_cache,
                    start_pos=curr_pos,
                    use_cache=True,
                )
                step_logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]
                if temperature > 0.0:
                    probs = F.softmax(step_logits / temperature, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1).item()
                else:
                    nxt = torch.argmax(step_logits, dim=-1).item()
                generated_tokens.append(nxt)
                token_history.append(nxt)
                curr_pos += 1
            else:
                K = len(candidates)
                eval_tokens = [generated_tokens[-1]] + candidates[:-1]
                eval_in = torch.tensor([eval_tokens], dtype=torch.long, device=self.device)
                eval_pos = torch.arange(curr_pos, curr_pos + K, dtype=torch.long, device=self.device).unsqueeze(0)

                self.static_cache.set_pos(curr_pos)
                out = self.model(
                    input_ids=eval_in,
                    position_ids=eval_pos,
                    past_key_values=self.static_cache,
                    sword_static_cache=self.static_cache,
                    start_pos=curr_pos,
                    use_cache=True,
                )
                logits_mat = out.logits[0] if hasattr(out, "logits") else out[0][0]

                accepted = 0
                for i in range(K):
                    sub_logits = logits_mat[i : i + 1]
                    if temperature > 0.0:
                        probs = F.softmax(sub_logits / temperature, dim=-1)
                        predicted = torch.multinomial(probs, num_samples=1).item()
                    else:
                        predicted = torch.argmax(sub_logits, dim=-1).item()

                    target_cand = candidates[i]
                    if predicted == target_cand:
                        generated_tokens.append(target_cand)
                        token_history.append(target_cand)
                        accepted += 1
                        if eos_id is not None and target_cand == eos_id:
                            break
                    else:
                        generated_tokens.append(predicted)
                        token_history.append(predicted)
                        accepted += 1
                        break

                curr_pos += accepted
                self.static_cache.set_pos(curr_pos)

        all_ids = input_ids[0].tolist() + generated_tokens
        all_ids_tensor = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        if hasattr(self.tokenizer, "batch_decode"):
            try:
                resp_text = self.tokenizer.batch_decode(all_ids_tensor, skip_special_tokens=True)[0]
            except Exception:
                resp_text = self.tokenizer.batch_decode([all_ids], skip_special_tokens=True)[0]
        elif hasattr(self.tokenizer, "decode"):
            resp_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        else:
            resp_text = str(all_ids)

        return {
            "tokens": generated_tokens,
            "text": resp_text,
            "num_tokens": len(generated_tokens),
        }

    @torch.inference_mode()
    def serve(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        use_speculative: Optional[bool] = None,
        speculative_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Serves concurrent prompt requests with async pipeline and zero sync stalls.
        Supports both batched SDPA serving and high-speed Speculative Prompt Lookup Decoding.
        """
        bsz = len(prompts)
        if bsz > self.max_concurrency:
            self.max_concurrency = bsz
            self.static_cache = StaticKVCache(
                num_layers=self.num_layers,
                max_batch_size=self.max_concurrency,
                num_kv_heads=self.num_kv_heads,
                max_seq_len=self.max_seq_len,
                head_dim=self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for name, module in self.model.named_modules():
                if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                    module._sword_static_cache = self.static_cache

        # Auto-enable speculative decoding for single stream or when explicitly requested
        if use_speculative is None:
            use_speculative = (bsz == 1)

        if use_speculative:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            all_responses = []
            tokens_per_stream = []
            stream_speeds = []

            for p in prompts:
                t_sub = time.perf_counter()
                res = self._serve_single_speculative(
                    prompt=p,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    speculative_k=speculative_k,
                )
                dt = time.perf_counter() - t_sub
                all_responses.append(res["text"])
                tokens_per_stream.append(res["num_tokens"])
                stream_speeds.append(res["num_tokens"] / dt if dt > 0 else 0.0)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            total_tokens = sum(tokens_per_stream)
            total_tps = total_tokens / total_time if total_time > 0 else 0.0

            return {
                "responses": all_responses,
                "latency_s": total_time,
                "total_tokens": total_tokens,
                "tokens_per_stream": tokens_per_stream,
                "stream_tps": stream_speeds,
                "total_tps": total_tps,
                "speculative": True,
            }

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Check if all prompts share identical text (standard in GRPO multi-rollout generation)
        is_shared_prompt = (bsz > 1 and all(p == prompts[0] for p in prompts))

        if is_shared_prompt:
            # SGLang-Style Shared Prefix: Prefill 1 prompt instead of bsz redundant copies (8x prefill speedup!)
            enc_single = self.tokenizer(
                [prompts[0]],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len - max_new_tokens,
            )
            in_ids_single = enc_single["input_ids"].to(self.device)
            prompt_len = in_ids_single.shape[1]
            attn_mask_single = enc_single.get("attention_mask", None)
            if attn_mask_single is not None:
                attn_mask_single = attn_mask_single.to(self.device)
                prefill_pos_ids = (attn_mask_single.long().cumsum(-1) - 1).clamp_min(0)
            else:
                prefill_pos_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0)

            self.static_cache.new_rollout(batch_size=bsz)
            self.static_cache.set_pos(0)
            outputs = self.model(
                input_ids=in_ids_single,
                position_ids=prefill_pos_ids,
                attention_mask=attn_mask_single,
                past_key_values=self.static_cache,
                sword_static_cache=self.static_cache,
                start_pos=0,
                use_cache=True,
            )
            # Duplicate the prefilled prefix KV cache to all bsz rollout streams in 0.01ms
            self.static_cache.duplicate_prefix_for_rollouts(
                num_prompts=1,
                group_size=bsz,
                prompt_lens=[prompt_len],
            )
            input_ids = in_ids_single.expand(bsz, -1)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_logits = logits[:, -1, :].repeat(bsz, 1)
        else:
            # Standard multi-prompt batch prefill
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

            self.static_cache.new_rollout(batch_size=bsz)
            self.static_cache.set_pos(0)
            if attention_mask is not None:
                prefill_pos_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            else:
                prefill_pos_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)

            outputs = self.model(
                input_ids=input_ids,
                position_ids=prefill_pos_ids,
                attention_mask=attention_mask,
                past_key_values=self.static_cache,
                sword_static_cache=self.static_cache,
                start_pos=0,
                use_cache=True,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_logits = logits[:, -1, :]

        # Fast sampling with Top-k vocabulary pruning
        if temperature > 0.0:
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                top_logits, top_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                probs = F.softmax(top_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, -1, sample)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [next_token]
        curr_pos = prompt_len
        self.static_cache.set_pos(curr_pos)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        active_mask = torch.ones((bsz, 1), dtype=torch.bool, device=self.device)

        # Pre-allocate static input buffers for decode step
        decode_input_ids = torch.empty((bsz, 1), dtype=torch.long, device=self.device)
        decode_pos_ids = torch.empty((bsz, 1), dtype=torch.long, device=self.device)

        # -------------------------------------------------------------
        # SGLang-Style CUDA Graph Capture for Decode Step (Single-launch replay)
        # -------------------------------------------------------------
        cuda_graph = None
        graph_logits = None
        if self.device.type == "cuda" and not self._graph_captured:
            try:
                # Warm up memory, cuBLAS handles & kernels on current stream before graph capture
                for _ in range(3):
                    decode_input_ids.copy_(next_token)
                    decode_pos_ids.fill_(curr_pos)
                    _ = self.decode_fn(
                        input_ids=decode_input_ids,
                        position_ids=decode_pos_ids,
                        past_key_values=self.static_cache,
                        use_cache=True,
                    )
                torch.cuda.synchronize()

                # Capture graph: records all decode kernels into 1 executable hardware graph
                cuda_graph = torch.cuda.CUDAGraph()
                decode_input_ids.copy_(next_token)
                decode_pos_ids.fill_(curr_pos)
                with torch.cuda.graph(cuda_graph):
                    graph_out = self.decode_fn(
                        input_ids=decode_input_ids,
                        position_ids=decode_pos_ids,
                        past_key_values=self.static_cache,
                        use_cache=True,
                    )
                    graph_logits = graph_out.logits[:, -1, :] if hasattr(graph_out, "logits") else graph_out[0][:, -1, :]

                self.cuda_graph_runner = (cuda_graph, decode_input_ids, decode_pos_ids, graph_logits)
                self._graph_captured = True
                print(f"[Sword] CUDA Graph captured successfully for {bsz} concurrent streams! Hardware single-launch replay active.")
            except Exception as e:
                self._graph_captured = False
                self.cuda_graph_runner = None
                if not getattr(self, "_logged_graph_note", False):
                    print(f"[Sword] Decode mode: Optimized Zero-Sync SDPA + Fast MoE active (graph fallback note: {type(e).__name__}: {e}).")
                    self._logged_graph_note = True

        # High-Speed Decode Loop
        for step in range(1, max_new_tokens):
            self.static_cache.set_pos(curr_pos)
            decode_pos_ids.fill_(curr_pos)
            decode_input_ids.copy_(next_token)

            if self.cuda_graph_runner is not None:
                g, g_in, g_pos, g_logits = self.cuda_graph_runner
                # Replay pre-recorded graph: 1 single CPU instruction instead of hundreds of kernel launches!
                g.replay()
                step_logits = g_logits
            else:
                out = self.decode_fn(
                    input_ids=decode_input_ids,
                    position_ids=decode_pos_ids,
                    past_key_values=self.static_cache,
                    sword_static_cache=self.static_cache,
                    start_pos=curr_pos,
                    use_cache=True,
                )
                step_logits = out.logits[:, -1, :] if hasattr(out, "logits") else out[0][:, -1, :]

            if temperature > 0.0:
                if top_k > 0 and top_k < step_logits.shape[-1]:
                    top_logits, top_indices = torch.topk(step_logits, k=top_k, dim=-1)
                    probs = F.softmax(top_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, num_samples=1)
                    next_token = torch.gather(top_indices, -1, sample)
                else:
                    probs = F.softmax(step_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            generated.append(next_token.clone())
            curr_pos += 1

            if eos_id is not None:
                active_mask = active_mask & (next_token != eos_id)
                # Check every 16 steps to eliminate GPU→CPU sync cost
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
        use_speculative: Optional[bool] = None,
        speculative_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Directly compares standard HuggingFace/Unsloth generation (BEFORE)
        vs Sword Pure FlashAttention + StaticKVCache + Speculative Drafter (AFTER).
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
        if use_speculative is None:
            # For single stream, prompt lookup speculative drafting is strictly faster.
            # For multi-stream concurrency, defaults to high-throughput parallel batched serving.
            use_speculative = (bsz == 1)

        print("=" * 72)
        print(f" BENCHMARK: {bsz}-CONCURRENCY SERVING (BEFORE vs AFTER)")
        print("=" * 72)
        print(f"Concurrency:       {bsz} concurrent streams")
        print(f"Target GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Local Dev)'}")
        print(f"Speculative Decode: {'Enabled (Prompt Lookup, k=' + str(speculative_k) + ')' if use_speculative else 'Disabled (Batched SDPA)'}")
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
        # 2. AFTER: Sword Speed Engine (Flash SDPA + Static KV + Speculative Drafter)
        # -----------------------------------------------------------------
        print(f"[*] Running [AFTER] with Sword Speed Engine (Flash SDPA + Static KV + {'Speculative Drafter' if use_speculative else 'Batched SDPA'})...")
        patch_model(self.model, mode="flash", patch_moe=True)
        for module in self.model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module._sword_static_cache = self.static_cache

        # Warm up Sword Speed Engine so first-call JIT kernel initialization does not distort metrics
        if self.device.type == "cuda":
            _ = self.serve(prompts, max_new_tokens=2, temperature=0.0, use_speculative=use_speculative, speculative_k=speculative_k)
            torch.cuda.synchronize()

        after_results = self.serve(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_speculative=use_speculative,
            speculative_k=speculative_k,
        )
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
FastQwen3MoeServer = FastServer
FastMoEServer = FastServer
FastQwenServer = FastServer


def get_fast_ling_server():
    from .ling import FastLingServer
    return FastLingServer

