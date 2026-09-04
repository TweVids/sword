import os
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .patcher import patch_qwen


def load_qwen_model(
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    dtype: Optional[torch.dtype] = None,
    use_unsloth: bool = True,
) -> Tuple[object, object]:
    """
    Loads Qwen (Qwen2, Qwen2.5, Qwen3, Qwen3.5) model using Unsloth (if available on GPU)
    or Transformers + BitsAndBytes quantization, and applies Sword's Pure FlashAttention patch.
    
    Args:
        model_name_or_path: HuggingFace model repo id or local checkpoint path
        load_in_4bit: Use 4-bit NF4 bitsandbytes quantization
        load_in_8bit: Use 8-bit bitsandbytes quantization
        device_map: Hardware placement ('auto', 'cuda', etc.)
        dtype: Compute precision (defaults to bfloat16 if GPU supported)
        use_unsloth: Whether to try Unsloth FastLanguageModel first on GPU
        
    Returns:
        Tuple of (patched_model, tokenizer)
    """
    if dtype is None:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None

    # Try Unsloth FastLanguageModel first if on GPU
    if use_unsloth and torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel
            print(f"[Sword] Loading {model_name_or_path} with Unsloth FastLanguageModel (4-bit={load_in_4bit})...")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name_or_path,
                max_seq_length=4096,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
        except Exception as e:
            print(f"[Sword] Unsloth load skipped ({e}). Falling back to Transformers + BitsAndBytes...")

    # Standard Transformers + BitsAndBytes fallback / direct loader
    if model is None:
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[Sword] Loading {model_name_or_path} via Transformers + BitsAndBytes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            device_map=device_map if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    # Patch with pure PyTorch FlashAttention
    model = patch_qwen(model)
    model.eval()

    return model, tokenizer
