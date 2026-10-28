"""Explicit templates override base/LoRA assets without mixing tokenizer caches."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer

import sword
from sword.chat_templates import read_chat_template, RenderedChatTokenizer
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, format_rollout_prompt
from sword.lora_source import prepare_merged_lora_base, resolve_lora_training_source
from sword.long_trainer import run_long_train
from test.test_judge_rule_context import native_thinking_tokenizer
from test.test_lora_source import local_source
from test.test_reward_stall import TinyPolicy, batch


def test_invalid_template_fails_before_lora_download(tmp_path):
    with patch("sword.lora_source.resolve_lora_training_source", side_effect=AssertionError("must not download")):
        with pytest.raises(FileNotFoundError):
            sword.train(test=False, model_lora_repo="user/repo", chat_template_path=str(tmp_path / "missing.jinja"))
    path = tmp_path / "bad.jinja"
    path.write_text("{% if broken %}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid chat template"):
        read_chat_template(path)
    path.write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_chat_template(path)


def test_relative_template_path_and_bom_are_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("chat_template.jinja")
    path.write_text("{{ bos_token }}custom", encoding="utf-8-sig")
    assert read_chat_template(str(path)) == "{{ bos_token }}custom"


def test_base_tokenizer_override_changes_rendering_and_reaches_fallback(tmp_path):
    path = tmp_path / "custom.jinja"
    path.write_text("{{ bos_token }}CUSTOM{% for message in messages %}{{ message.content }}{% endfor %}", encoding="utf-8")
    tokenizer = native_thinking_tokenizer(False)
    bridge = GRPOBridge(GRPOBridgeConfig(device="cpu", chat_template_path=str(path)))
    try:
        bridge.initialize_trainer(model=TinyPolicy(), tokenizer=tokenizer)
        rendered, formatted = format_rollout_prompt(bridge.tokenizer, "task")
        assert formatted and "CUSTOMtask" in rendered
        assert bridge.fallback_engine.tokenizer.chat_template == path.read_text()
    finally:
        bridge.weight_sync.cleanup()


def test_serving_template_does_not_insert_bos_twice():
    tokenizer = native_thinking_tokenizer(False)
    from tokenizers.processors import TemplateProcessing
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 0)])
    tokenizer.chat_template = "{{ bos_token }} p"
    prompt = format_rollout_prompt(tokenizer, "task")[0]
    encoded = RenderedChatTokenizer(tokenizer)(prompt)["input_ids"]
    assert encoded == [0, 2]


def test_lora_override_uses_distinct_merged_cache_and_is_serialized(local_source, tmp_path):
    source = resolve_lora_training_source(local_source.repo, 900)
    override = local_source.template.replace("Effort:", "CUSTOM Effort:")
    path = tmp_path / "custom.jinja"
    path.write_text(override, encoding="utf-8")
    cache = str(tmp_path / "merged")
    with patch("torch.cuda.is_available", return_value=False):
        original = prepare_merged_lora_base(source, cache)
    bridge = GRPOBridge(GRPOBridgeConfig(device="cpu", lora_training_source=source,
                                        chat_template_path=str(path), merged_base_cache_dir=cache))
    try:
        with patch("torch.cuda.is_available", return_value=False):
            bridge.initialize_trainer()
        assert bridge.config.model_name != original
        assert bridge.tokenizer.chat_template == override
        assert AutoTokenizer.from_pretrained(bridge.config.model_name).chat_template == override
        assert source.chat_template == local_source.template
        assert "CUSTOM Effort:" in format_rollout_prompt(bridge.tokenizer, "task")[0]
    finally:
        bridge.weight_sync.cleanup()


def test_public_training_saves_override_and_template_changes_are_new_work(tmp_path):
    template_path = tmp_path / "custom.jinja"
    template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"
    results = []
    def generate(bridge, prompts):
        result = batch()
        for index, item in enumerate(result.items):
            item.response_text = f"Final Answer: {4 + index}"
        return result
    for prefix in ("FIRST", "SECOND"):
        template_path.write_text(prefix + template, encoding="utf-8")
        with patch.object(GRPOBridge, "initialize_vllm"), \
             patch.object(GRPOBridge, "generate_rollouts", autospec=True, side_effect=generate):
            result = sword.train(test=False, model="mock", dataset=[{"problem": "2 + 2?", "solution": "4"}],
                                 b_model=False, chat_template_path=str(template_path),
                                 trainer_model=TinyPolicy(), tokenizer=native_thinking_tokenizer(False),
                                 is_save_and_upload_checkpoint=False, show_detail_logs=False,
                                 Argument={"steps": 1, "concurrency_g": 2, "max_seq_len": 256,
                                           "max_new_tokens": 1, "output_dir": str(tmp_path / "checkpoints")})
        assert result["status"] == "completed"
        assert (Path(result["checkpoint_dir"]) / "chat_template.jinja").read_text(encoding="utf-8") == prefix + template
        results.append(result)
    assert results[0]["dataset_sha256"] != results[1]["dataset_sha256"]
