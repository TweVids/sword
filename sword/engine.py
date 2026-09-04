import time
from typing import List, Optional, Union, Tuple
import torch
import torch.nn.functional as F

from .model import FastTransformerModel, FastTransformerConfig
from .kv_cache import StaticKVCache


class SpeedEngine:
    """
    High-Throughput Pure-PyTorch Generation Engine.
    
    Optimized for target concurrency of 4+ streams on modern architectures (e.g. Blackwell).
    Features:
    - Zero-allocation Static KV Caching
    - Native PyTorch SDPA FlashAttention dispatch
    - Support for torch.compile(mode="reduce-overhead") CUDA graph acceleration
    - Streamlined decode loop avoiding synchronization overhead
    """
    def __init__(
        self,
        model: FastTransformerModel,
        max_batch_size: int = 4,
        max_seq_len: int = 2048,
        compile_decode: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Initialize pre-allocated KV Cache
        config = self.model.config
        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.kv_cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            max_batch_size=max_batch_size,
            num_kv_heads=config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.compile_decode = compile_decode
        if compile_decode and self.device.type == "cuda":
            print("[SpeedEngine] Compiling decode step with mode='reduce-overhead' (CUDA graphs)...")
            try:
                self.decode_forward = torch.compile(self.model.forward, mode="reduce-overhead")
            except Exception as e:
                print(f"[SpeedEngine] Warning: compile failed ({e}), falling back to eager mode.")
                self.decode_forward = self.model.forward
        else:
            self.decode_forward = self.model.forward

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate completions for a batch of prompts (concurrency = batch size).
        
        Args:
            input_ids: [batch_size, prompt_len]
            max_new_tokens: maximum tokens to generate
            temperature: 0.0 for greedy, >0.0 for sampling
            top_k: top-k sampling threshold
            eos_token_id: optional end-of-sequence token id
            
        Returns:
            Tuple of (generated_ids [batch_size, prompt_len + generated_len], stats_dict)
        """
        bsz, prompt_len = input_ids.shape
        assert bsz <= self.max_batch_size, f"Batch size {bsz} exceeds max {self.max_batch_size}"
        assert prompt_len + max_new_tokens <= self.max_seq_len, "Total sequence length exceeds max_seq_len"

        input_ids = input_ids.to(self.device)
        self.kv_cache.reset(list(range(bsz)))

        # Synchronize for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # ==========================================
        # 1. Prefill Phase
        # ==========================================
        prefill_start = time.perf_counter()
        logits = self.model(
            input_ids=input_ids,
            start_pos=0,
            kv_cache=self.kv_cache,
            return_logits=True,
        )

        # Extract logits for the next token (last position)
        next_token_logits = logits[:, -1, :]
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start

        # Storage for output tokens
        generated_tokens = [next_token]
        curr_pos = prompt_len
        active_mask = torch.ones((bsz, 1), dtype=torch.bool, device=self.device)

        # ==========================================
        # 2. Fast Decode Phase
        # ==========================================
        decode_start = time.perf_counter()
        tokens_generated = 1

        for step in range(1, max_new_tokens):
            curr_input = next_token  # [bsz, 1]

            logits = self.decode_forward(
                input_ids=curr_input,
                start_pos=curr_pos,
                kv_cache=self.kv_cache,
                return_logits=True,
            )

            next_token_logits = logits[:, 0, :]
            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                is_eos = (next_token == eos_token_id)
                active_mask = active_mask & (~is_eos)
                if not active_mask.any():
                    generated_tokens.append(next_token)
                    tokens_generated += 1
                    break

            generated_tokens.append(next_token)
            curr_pos += 1
            tokens_generated += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        decode_time = time.perf_counter() - decode_start

        all_generated = torch.cat([input_ids] + generated_tokens, dim=1)
        total_tokens = tokens_generated * bsz
        decode_tps = (tokens_generated * bsz) / decode_time if decode_time > 0 else 0.0
        total_tps = total_tokens / total_time if total_time > 0 else 0.0

        stats = {
            "batch_size": bsz,
            "tokens_per_stream": tokens_generated,
            "total_tokens": total_tokens,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "total_time_s": total_time,
            "decode_tps": decode_tps,
            "total_tps": total_tps,
        }

        return all_generated, stats
