"""Fixed checkpoint selection, safe merging, and GRPO/vLLM policy consistency."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import sword
from sword.checkpoint import CheckpointManager
from sword.grpo_bridge import GRPOBridge, GRPOBridgeConfig, tokenize_rollout_prompt
from sword.lora_source import (
    SOURCE_MANIFEST, normalize_lora_checkpoint, prepare_merged_lora_base, resolve_lora_training_source,
)
from test.test_judge_rule_context import native_thinking_tokenizer


@pytest.fixture
def local_source(tmp_path):
    torch.manual_seed(93)
    base_dir, repo = tmp_path / "base", tmp_path / "source"
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
        attention_dropout=0, tie_word_embeddings=False,
    ))
    model.save_pretrained(base_dir)
    tokenizer = native_thinking_tokenizer(False)
    template = (Path(__file__).parents[1] / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer.chat_template = template
    tokenizer.save_pretrained(base_dir)
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.data.fill_(0.15)
        if "lora_B" in name:
            param.data.fill_(0.1)
    for step in (100, 900):
        model.save_pretrained(repo / f"checkpoint-{step}")
    repo.mkdir(exist_ok=True)
    (repo / "chat_template.jinja").write_text(template, encoding="utf-8")
    shutil.copy2(repo / "checkpoint-900" / "adapter_config.json", repo / "adapter_config.json")
    # PEFT's model.name_or_path is empty for a from_config fixture; establish its real base.
    for config in repo.rglob("adapter_config.json"):
        data = json.loads(config.read_text())
        data["base_model_name_or_path"] = str(base_dir)
        config.write_text(json.dumps(data), encoding="utf-8")
    model.eval()
    with torch.no_grad():
        expected_logits = model(torch.tensor([[2, 4, 5]])).logits.clone()
    return SimpleNamespace(base=str(base_dir), repo=str(repo), tokenizer=tokenizer,
                           expected_logits=expected_logits, template=template)


@pytest.mark.parametrize("checkpoint", [900, "900", "checkpoint-900", "latest"])
def test_specific_checkpoint_and_base_are_resolved(local_source, checkpoint):
    source = resolve_lora_training_source(local_source.repo, checkpoint)
    assert source.base_model == local_source.base
    assert source.manifest["adapters"][-1]["checkpoint"] == "checkpoint-900"
    assert source.chat_template == local_source.template
    assert Path(source.adapters[-1]).name == "checkpoint-900"


def test_nested_duplicate_steps_require_full_folder(local_source):
    repo = Path(local_source.repo)
    shutil.copytree(repo / "checkpoint-900", repo / "run2" / "checkpoint-900")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_lora_training_source(str(repo), 900)
    selected = resolve_lora_training_source(str(repo), "run2/checkpoint-900")
    assert selected.manifest["adapters"][-1]["checkpoint"] == "run2/checkpoint-900"


def test_missing_checkpoint_and_conflicting_base_never_fall_back(local_source):
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_lora_training_source(local_source.repo, 800)
    with pytest.raises(FileNotFoundError, match="root"):
        resolve_lora_training_source(local_source.repo, "root")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_lora_training_source(local_source.repo, 900, base_model="different/base")


@pytest.mark.parametrize("checkpoint", ["../checkpoint-900", "/checkpoint-900", "run/../../x", "C:\\x", "C:/x", True, []])
def test_checkpoint_paths_cannot_escape_the_repo(checkpoint):
    with pytest.raises(ValueError):
        normalize_lora_checkpoint(checkpoint)


def test_checkpoint_can_borrow_root_config_without_modifying_source(local_source, tmp_path):
    original = Path(local_source.repo) / "checkpoint-900"
    (original / "adapter_config.json").unlink()
    source = resolve_lora_training_source(local_source.repo, 900, source_cache_dir=str(tmp_path / "staged"))
    assert not (original / "adapter_config.json").exists()
    assert Path(source.adapters[-1], "adapter_config.json").is_file()
    assert Path(source.adapters[-1], "adapter_model.safetensors").is_file()
    assert source.base_model == local_source.base


def test_remote_checkpoint_download_is_pinned_and_excludes_other_weights(local_source):
    root = Path(local_source.repo)
    files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    api = MagicMock()
    api.model_info.return_value.sha = "fixed-revision"
    api.list_repo_files.return_value = files + ["checkpoint-900/optimizer.pt", "checkpoint-900/trainer_state.json"]
    with patch("huggingface_hub.HfApi", return_value=api) as constructor, \
         patch("huggingface_hub.snapshot_download", return_value=str(root)) as download:
        source = resolve_lora_training_source("user/source", 900, token="private-token")
    constructor.assert_called_with(token="private-token")
    api.list_repo_files.assert_called_once_with("user/source", revision="fixed-revision")
    options = download.call_args.kwargs
    assert options["revision"] == "fixed-revision" and options["token"] == "private-token"
    assert "checkpoint-900/adapter_model.safetensors" in options["allow_patterns"]
    assert not any("checkpoint-100" in value or "optimizer" in value for value in options["allow_patterns"])
    assert source.manifest["adapters"][-1]["revision"] == "fixed-revision"


def test_real_merge_matches_source_policy_and_reuses_the_full_base(local_source, tmp_path):
    source = resolve_lora_training_source(local_source.repo, 900)
    with patch("torch.cuda.is_available", return_value=False):
        merged = prepare_merged_lora_base(source, str(tmp_path / "merged"))
    model = AutoModelForCausalLM.from_pretrained(merged).eval()
    assert not any("lora_" in name for name, _ in model.named_parameters())
    with torch.no_grad():
        actual = model(torch.tensor([[2, 4, 5]])).logits
    assert torch.allclose(actual, local_source.expected_logits, atol=2e-6, rtol=1e-5)
    assert AutoTokenizer.from_pretrained(merged).chat_template == local_source.template
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("must reuse cache")):
        assert prepare_merged_lora_base(source, str(tmp_path / "merged")) == merged


def test_merge_failure_stops_before_trainer_or_rollout_loading(local_source, tmp_path):
    source = resolve_lora_training_source(local_source.repo, 900)
    bridge = GRPOBridge(GRPOBridgeConfig(lora_training_source=source, device="cpu",
                                        merged_base_cache_dir=str(tmp_path / "merged")))
    try:
        with patch("torch.cuda.is_available", return_value=False), \
             patch("peft.PeftModel.from_pretrained", side_effect=RuntimeError("incompatible adapter")):
            with pytest.raises(RuntimeError, match="will not use the original base"):
                bridge.initialize_trainer()
        assert bridge.trainer_model is None and bridge.vllm_engine is None
        assert not list((tmp_path / "merged").iterdir())
    finally:
        bridge.weight_sync.cleanup()


def test_vllm_loads_the_merged_base_and_fresh_adapter_stays_in_sync(local_source, tmp_path):
    source = resolve_lora_training_source(local_source.repo, 900)
    bridge = GRPOBridge(GRPOBridgeConfig(
        lora_training_source=source, model_name=source.base_model, device="cpu", concurrency_g=2,
        max_model_len=128, max_new_tokens=1, lora_r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        merged_base_cache_dir=str(tmp_path / "merged"), adapter_sync_dir=str(tmp_path / "sync"),
        entropy_coeff=0, offload_trainer_during_rollout=True, require_vllm=True,
    ))
    loads, versions = [], []

    class CpuRolloutEngine:
        def __init__(self, **options):
            loads.append(options)
            self.base = options["model"]
            assert Path(self.base, SOURCE_MANIFEST).is_file()

        def sleep(self, **kwargs):
            pass

        def wake_up(self, **kwargs):
            pass

        def generate(self, prompts, sampling, lora_request):
            versions.append(lora_request.lora_int_id)
            model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(self.base), lora_request.lora_path).eval()
            outputs = []
            bridge.trainer_model.eval()
            for index, prompt in enumerate(prompts):
                ids = prompt["prompt_token_ids"]
                inputs = torch.tensor([ids])
                with torch.no_grad():
                    logits = model(inputs).logits[:, -1, :]
                    trainer_logits = bridge.trainer_model(inputs).logits[:, -1, :]
                assert torch.allclose(logits, trainer_logits, atol=2e-6, rtol=1e-5)
                token_id = 5 + index
                outputs.append(SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(
                    text="4", token_ids=[token_id], finish_reason="stop",
                    logprobs=[{token_id: SimpleNamespace(logprob=logits.log_softmax(-1)[0, token_id].item())}],
                )]))
            return outputs

    fake_vllm = SimpleNamespace(LLM=CpuRolloutEngine, SamplingParams=lambda **options: options)
    modules = {"vllm": fake_vllm, "vllm.lora.request": SimpleNamespace(LoRARequest=lambda **options: SimpleNamespace(**options))}
    try:
        with patch("torch.cuda.is_available", return_value=False), patch("sword.grpo_bridge.HAS_UNSLOTH", False), \
             patch.dict(sys.modules, modules):
            bridge.initialize_trainer()
            merged_path = bridge.config.model_name
            assert merged_path != source.base_model
            initial = {name: param.detach().clone() for name, param in bridge.trainer_model.named_parameters()}
            trainable = {name: param for name, param in bridge.trainer_model.named_parameters() if param.requires_grad}
            assert trainable and all("lora_" in name for name in trainable)
            assert all(not param.detach().count_nonzero() for name, param in trainable.items() if "lora_B" in name)
            assert bridge.initialize_vllm()
            assert loads[0]["model"] == merged_path and bridge.weight_sync.base_model_name == merged_path
            batch = bridge.generate_rollouts(["p"])
            for index, item in enumerate(batch.items):
                item.reward = float(index)
            bridge.compute_advantages(batch)
            metrics = bridge.train_step(batch, torch.optim.SGD(trainable.values(), lr=0.1))
            assert metrics["grad_norm"] > 0 and not metrics["optimizer_step_skipped"]
            assert any(not torch.equal(initial[name], param) for name, param in trainable.items())
            assert all(torch.equal(initial[name], param) for name, param in bridge.trainer_model.named_parameters() if not param.requires_grad)
            bridge.generate_rollouts(["p"])
            assert versions == [1, 2] and len(loads) == 1
    finally:
        bridge.weight_sync.cleanup()


def test_new_checkpoint_reconstructs_its_parent_base_before_merging_again(local_source, tmp_path):
    source = resolve_lora_training_source(local_source.repo, 900)
    with patch("torch.cuda.is_available", return_value=False):
        merged = prepare_merged_lora_base(source, str(tmp_path / "merged"))
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(merged), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.fill_(0.025)
    model.eval()
    with torch.no_grad():
        expected = model(torch.tensor([[2, 4, 5]])).logits
    manifest = json.loads(Path(merged, SOURCE_MANIFEST).read_text())
    manager = CheckpointManager(output_dir=str(tmp_path / "new_repo"), is_save_and_upload_checkpoint=False)
    checkpoint = manager.save_checkpoint(5, model, base_model_name=merged,
                                         source_lora_manifest=manifest, tokenizer=local_source.tokenizer)
    assert Path(checkpoint, SOURCE_MANIFEST).is_file()
    # A later machine has no local merged base; the source manifest must be sufficient.
    config_path = Path(checkpoint, "adapter_config.json")
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "/unavailable/old/merged/base"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    next_source = resolve_lora_training_source(str(tmp_path / "new_repo"), 5)
    assert next_source.base_model == local_source.base and len(next_source.adapters) == 2
    with patch("torch.cuda.is_available", return_value=False):
        next_base = prepare_merged_lora_base(next_source, str(tmp_path / "next_merged"))
    reconstructed = AutoModelForCausalLM.from_pretrained(next_base).eval()
    with torch.no_grad():
        actual = reconstructed(torch.tensor([[2, 4, 5]])).logits
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("prep", [False, True])
def test_public_api_resolves_lora_before_setup_and_passes_selected_source(local_source, prep):
    with patch("sword.long_trainer.run_long_train", return_value={"status": "completed"}) as train, \
         patch("sword.train_entry._run_prep_and_logs", return_value={"ok": True}) as diagnostics:
        sword.train(model_lora_repo=local_source.repo, train_from_lora_checkpoint=900,
                    prep_and_logs=prep, b_model=False, load_in_8bit=False)
    call = diagnostics.call_args if prep else train.call_args
    assert call.kwargs["model"] == local_source.base
    assert call.kwargs["model_lora_repo"] == local_source.repo
    assert call.kwargs["_lora_training_source"].manifest["adapters"][-1]["checkpoint"] == "checkpoint-900"


def test_checkpoint_argument_without_repo_is_rejected():
    with pytest.raises(ValueError, match="requires model_lora_repo"):
        sword.train(train_from_lora_checkpoint=900)


def test_full_train_saves_provenance_and_scopes_completion_to_the_source(local_source, tmp_path):
    from sword.grpo_bridge import FallbackRolloutEngine

    def generate(engine, prompts, **options):
        results = []
        for index, prompt in enumerate(prompts):
            inputs, _ = tokenize_rollout_prompt(engine.tokenizer, prompt, effort=options["effort"])
            ids = inputs["input_ids"][0].tolist()
            token_id = 5 + index
            with torch.no_grad():
                logprob = engine.model(**inputs).logits[0, -1].log_softmax(-1)[token_id].item()
            results.append({"prompt_token_ids": ids, "response_token_ids": [token_id],
                            "response_text": "Final Answer: " + ("4" if index == 0 else "9"),
                            "rollout_logprobs": [logprob], "is_truncated": False})
        return results

    options = dict(
        test=False, model_lora_repo=local_source.repo, b_model=False,
        load_in_8bit=False, is_save_and_upload_checkpoint=False, show_detail_logs=False,
        dataset=[{"problem": "p", "solution": "4", "effort": "high"}],
        Argument={"steps": 1, "concurrency_g": 2, "max_seq_len": 128, "max_new_tokens": 1,
                  "output_dir": str(tmp_path / "train")},
    )
    with patch("torch.cuda.is_available", return_value=False), patch("sword.grpo_bridge.HAS_UNSLOTH", False), \
         patch.object(GRPOBridge, "initialize_vllm", return_value=False), \
         patch.object(FallbackRolloutEngine, "generate", autospec=True, side_effect=generate):
        first = sword.train(train_from_lora_checkpoint=900, **options)
        saved_first = tmp_path / "first_checkpoint"
        shutil.copytree(first["checkpoint_dir"], saved_first)
        second = sword.train(train_from_lora_checkpoint=100, **options)
        repeated = sword.train(train_from_lora_checkpoint=100, **options)
        with pytest.raises(ValueError, match="different source LoRA"):
            sword.train(train_from_lora_checkpoint=100, resume_from_checkpoint=str(saved_first),
                        allow_retest=True, **options)
    assert first["status"] == second["status"] == "completed" and repeated["status"] == "already_trained"
    assert first["optimizer_steps"] == second["optimizer_steps"] == 1
    assert first["dataset_sha256"] != second["dataset_sha256"]
    checkpoint = Path(second["checkpoint_dir"])
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    manifest = json.loads((checkpoint / SOURCE_MANIFEST).read_text())
    assert config["base_model_name_or_path"] == second["training_base_model"]
    assert manifest == second["source_lora"] and manifest["adapters"][-1]["checkpoint"] == "checkpoint-100"
    assert (checkpoint / "chat_template.jinja").read_text(encoding="utf-8") == local_source.template
