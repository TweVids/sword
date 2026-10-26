"""Multimodal target selection must connect text GRPO to language adapters."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sword.grpo_bridge import (
    DualCorrectionGRPOLoss, GRPOBridge, GRPOBridgeConfig, RolloutBatch,
    RolloutItem, peft_compatible_target_names,
)


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def gemma_model(quantized=False):
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    torch.manual_seed(31)
    text = Gemma4TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, global_head_dim=8,
        max_position_embeddings=32, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=32, layer_types=["full_attention"],
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    )
    vision = Gemma4VisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        patch_size=2, pooling_kernel_size=1, position_embedding_size=16,
        use_clipped_linears=True,
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision))
    if quantized:
        import bitsandbytes as bnb
        from peft import prepare_model_for_kbit_training
        for name, original in list(model.named_modules()):
            parts = name.split(".")
            if not isinstance(original, torch.nn.Linear) or not (
                parts[-1] in TARGETS or parts[-1] == "linear" and parts[-2] in TARGETS
            ):
                continue
            layer = bnb.nn.Linear8bitLt(
                original.in_features, original.out_features,
                bias=original.bias is not None, has_fp16_weights=False,
            )
            layer.weight = bnb.nn.Int8Params(
                original.weight.detach().clone(), requires_grad=False, has_fp16_weights=False,
            )
            if original.bias is not None:
                layer.bias = torch.nn.Parameter(original.bias.detach().clone(), requires_grad=False)
            setattr(model.get_submodule(".".join(parts[:-1])), parts[-1], layer.to("cpu"))
        model.is_loaded_in_8bit = True
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model


def test_mixed_gemma_targets_retain_every_direct_projection_and_wrapper():
    from peft import LoraConfig, get_peft_model
    model = gemma_model()
    targets = peft_compatible_target_names(model, TARGETS)
    assert len(targets) == 14
    assert sum("language_model" in name for name in targets) == 7
    assert sum("vision_tower" in name and name.endswith(".linear") for name in targets) == 7
    model = get_peft_model(model, LoraConfig(r=2, target_modules=targets, task_type="CAUSAL_LM"))
    assert sum("lora_" in name for name, _ in model.named_parameters()) == 28


def test_wrapper_coverage_cannot_silently_drop_an_unsupported_sibling():
    model = torch.nn.Module()
    model.good = torch.nn.Module()
    model.good.q_proj = torch.nn.Module()
    model.good.q_proj.linear = torch.nn.Linear(4, 4)
    model.bad = torch.nn.Module()
    model.bad.q_proj = torch.nn.Module()
    with pytest.raises(ValueError, match=r"bad.q_proj"):
        peft_compatible_target_names(model, ["q_proj"])
    assert not any("lora_" in name for name, _ in model.named_parameters())


def test_text_scope_rejects_explicit_vision_only_targets():
    model = gemma_model()
    with pytest.raises(ValueError, match="only outside the text language decoder"):
        peft_compatible_target_names(
            model, ["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear"], text_only=True,
        )


def test_legacy_vision_only_targets_reproduce_missing_gradients_and_fail_startup():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(gemma_model(quantized=True), LoraConfig(
        r=2, target_modules=[target + ".linear" for target in TARGETS], task_type="CAUSAL_LM",
    ))
    model.enable_input_require_grads()
    inputs = torch.tensor([[2, 4, 5], [2, 4, 6]])
    loss, _ = DualCorrectionGRPOLoss(entropy_coeff=0)(
        model, inputs, torch.ones_like(inputs), [2, 2], torch.tensor([-1., 1.]),
    )
    assert loss.requires_grad  # embedding hooks alone do not prove adapter connectivity
    loss.backward()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert len(trainable) == 14
    assert all("vision_tower" in name and param.grad is None for name, param in trainable)
    bridge = GRPOBridge(GRPOBridgeConfig(device="cpu"))
    try:
        with pytest.raises(RuntimeError, match="No trainable parameters in the text language decoder"):
            bridge.initialize_trainer(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
    finally:
        bridge.weight_sync.cleanup()


@pytest.mark.parametrize("quantized", [False, True])
def test_real_gemma_text_grpo_updates_language_lora_and_exports_it(quantized, tmp_path):
    from safetensors.torch import load_file
    model = gemma_model(quantized=quantized)
    config = GRPOBridgeConfig(
        model_name="local-gemma-fixture", device="cpu", concurrency_g=2,
        micro_batch_size=1, entropy_coeff=0, target_modules=TARGETS,
        lora_r=2, lora_alpha=4, load_in_8bit=quantized, adapter_sync_dir=str(tmp_path),
    )
    bridge = GRPOBridge(config)
    try:
        # Only model download/tokenizer construction are replaced. The actual
        # Gemma, quantization, PEFT setup, checkpointed backward and export run.
        with patch("sword.grpo_bridge.HAS_UNSLOTH", False), \
             patch("sword.grpo_bridge.AutoModelForCausalLM.from_pretrained", return_value=model), \
             patch("sword.grpo_bridge.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(pad_token_id=0)):
            bridge.initialize_trainer()
        targets = bridge.config.target_modules
        assert len(targets) == 7 and all(name.startswith("model.language_model.") for name in targets)
        assert bridge.weight_sync.target_modules == targets
        model = bridge.trainer_model
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        assert len(trainable) == 14 and all("language_model" in name for name in trainable)
        before = {name: param.detach().clone() for name, param in trainable.items()}
        batch = RolloutBatch(items=[
            RolloutItem("task", [2, 4], "bad", [5], [-math.log(32)], reward=0),
            RolloutItem("task", [2, 4], "good", [6], [-math.log(32)], reward=1),
        ], concurrency_g=2)
        bridge.compute_advantages(batch)
        metrics = bridge.train_step(batch, torch.optim.SGD(trainable.values(), lr=0.1))
        assert metrics["grad_norm"] > 0 and not metrics["optimizer_step_skipped"]
        assert metrics["tensors_with_grad"] == 14 and metrics["tensors_with_nonzero_grad"] > 0
        assert any(not torch.equal(before[name], param) for name, param in trainable.items())
        exported = load_file(str(bridge.weight_sync.current_adapter_path) + "/adapter_model.safetensors")
        assert len(exported) == 14 and all("language_model" in name for name in exported)
        assert not any("vision_tower" in name for name in exported)
    finally:
        bridge.weight_sync.cleanup()
