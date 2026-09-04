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
    max_seq_length: int = 8192,
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
        max_seq_length: Maximum sequence context length (default 8192)
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
            print(f"[Sword] Loading {model_name_or_path} with Unsloth FastLanguageModel (4-bit={load_in_4bit}, max_seq={max_seq_length})...")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name_or_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
        except Exception as e:
            err_msg = str(e)
            print(f"[Sword] Unsloth load skipped ({err_msg}). Falling back to Transformers + BitsAndBytes...")
            if "torchvision" in err_msg.lower():
                print("[Sword] Tip: Run `pip install torchvision` to enable Unsloth vision/multimodal support.")

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
            # Only pass torch_dtype for non-quantized loads; quantization_config controls dtype for 4/8-bit
            torch_dtype=dtype if quant_config is None else None,
            trust_remote_code=True,
        )

    # Patch with pure PyTorch FlashAttention
    model = patch_qwen(model)
    model.eval()

    return model, tokenizer


def load_moe_model(
    model_name_or_path: str = "tencent/Hy-MT2-30B-A3B-FP8",
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    max_seq_length: int = 8192,
    patch_sword: bool = True,
    attn_mode: str = "flash",
) -> Tuple[object, object]:
    """
    Loads MoE (Mixture of Experts) models, including FP8 pre-quantized models
    such as tencent/Hy-MT2-30B-A3B-FP8, Qwen2-MoE, DeepSeek, etc.
    
    Directly leverages native FP8 Tensor Cores without bitsandbytes overhead.
    Applies Sword's pure FlashAttention SDPA speed engine for high-throughput serving.
    
    Args:
        model_name_or_path: HuggingFace model repo id or local checkpoint path
        device_map: Hardware placement ('auto', 'cuda', etc.)
        torch_dtype: Compute precision (defaults to 'auto' for FP8 safetensors)
        max_seq_length: Maximum sequence context length
        patch_sword: Whether to automatically inject Sword FlashAttention SDPA
        attn_mode: 'flash' or 'vanilla'
        
    Returns:
        Tuple of (patched_model, tokenizer)
    """
    print(f"\n[Sword] Loading MoE model: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Sword] Loading weights with native FP8/Tensor-Core acceleration (device_map='{device_map}')...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype or "auto",
        trust_remote_code=True,
    )

    if patch_sword:
        from .patcher import patch_model
        model = patch_model(model, mode=attn_mode)

    model.eval()
    print(f"[Sword] MoE model ready for serving.")
    return model, tokenizer
