"""Storage and inference doubles for task orchestration tests; no GPU or Hub access."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace


class Parameter:
    requires_grad = True
    def __init__(self, value):
        self.value = value
    def detach(self):
        return self
    def cpu(self):
        return self.value


class StorageModel:
    def __init__(self, value=1.0):
        self.weight = Parameter(value)
    def named_parameters(self):
        return [("weight", self.weight)]
    def parameters(self):
        return [self.weight]
    def state_dict(self):
        return {"weight": self.weight.value}
    def load_state_dict(self, state, strict=False):
        self.weight.value = state["weight"]


class StorageOptimizer:
    def __init__(self, parameters=(), lr=0.01):
        self.state = {"lr": lr, "momentum": [1.0, 2.0]}
        self.fail_restore = False
    def state_dict(self):
        return copy.deepcopy(self.state)
    def load_state_dict(self, state):
        if self.fail_restore:
            raise RuntimeError("incompatible optimizer")
        self.state = copy.deepcopy(state)
    def zero_grad(self, **kwargs):
        pass


class StorageTorch:
    class OutOfMemoryError(RuntimeError):
        pass
    def __init__(self):
        self.rng = [1, 2, 3]
        self.cuda = SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False,
                                    OutOfMemoryError=self.OutOfMemoryError)
        self.optim = SimpleNamespace(AdamW=StorageOptimizer, Optimizer=StorageOptimizer)
        self.nn = SimpleNamespace(Module=object)
    def get_rng_state(self):
        return list(self.rng)
    def set_rng_state(self, state):
        self.rng = list(state)
    def save(self, value, path):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
    def load(self, path, **kwargs):
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict) and "python" in value:
            version, state, gaussian = value["python"]
            value["python"] = (version, tuple(state), gaussian)
        return value


class Hub:
    def __init__(self):
        self.files, self.uploads = {}, []
        self.offline = False
    def create_repo(self, **kwargs):
        pass
    def upload_folder(self, *, folder_path, path_in_repo, **kwargs):
        self.uploads.append(path_in_repo)
        if self.offline:
            raise OSError("hub offline")
        folder = Path(folder_path)
        assert (folder / "checkpoint_manifest.json").is_file()
        for path in folder.rglob("*"):
            if path.is_file():
                self.files[path_in_repo + "/" + path.relative_to(folder).as_posix()] = path.read_bytes()
    def list_repo_files(self, **kwargs):
        return sorted(self.files)
    def download(self, *, filename, local_dir, **kwargs):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


class Score:
    def __init__(self, truth=0.0, total_score=0.7, cheating=False, judge_source="disabled"):
        self.truth, self.total_score, self.cheating = truth, total_score, cheating
        self.judge_source = judge_source
        self.judge_scores, self.judge_response = {}, ""
    def to_dict(self):
        return {"truth": self.truth, "total_score": self.total_score, "cheating": self.cheating}


class Scorer:
    def __init__(self, **kwargs):
        pass
    def score(self, **kwargs):
        return Score(truth=float(kwargs["candidate_response"] == "correct"))


class Bridge:
    instances = []
    answers = {}
    oom_attempts = 0
    def __init__(self, config):
        self.config = config
        self.source_lora_manifest = None
        self.vllm_engine, self.vllm_initialization_error = None, None
        self.weight_sync = SimpleNamespace(cleanup=lambda: None)
        self.vram_mgr = SimpleNamespace(enter_score_phase=lambda engine: None)
        self.loss_fn = SimpleNamespace(chunk_size=8)
        self.generation_callback = None
        self.global_step = 0
        self._reward_ema, self._reward_ema_var = None, 1.0
        self.prompts, self.observed_baselines = [], []
        self.train_calls = 0
        self.instances.append(self)
    def initialize_trainer(self, **kwargs):
        self.trainer_model = kwargs.get("model") or StorageModel()
        self.tokenizer = kwargs.get("tokenizer") or SimpleNamespace()
    def initialize_vllm(self):
        pass
    def generate_rollouts(self, prompts):
        self.prompts.extend(prompts)
        self.observed_baselines.append((self._reward_ema, self._reward_ema_var))
        response = self.answers.get(prompts[0], "wrong")
        return SimpleNamespace(items=[SimpleNamespace(
            prompt_text=prompts[0], prompt_token_ids=[1], response_text=response,
            response_token_ids=[2], rollout_logprobs=[-0.5], group_id=0,
            is_truncated=False, followup_turns=[], reward=0.0, advantage=0.0,
        ) for _ in range(self.config.concurrency_g)])
    def compute_advantages(self, batch):
        self._reward_ema = sum(item.reward for item in batch.items) / len(batch.items)
        self._reward_ema_var = 0.25
    def train_step(self, batch, optimizer):
        self.train_calls += 1
        if self.train_calls <= self.oom_attempts:
            raise StorageTorch.OutOfMemoryError("CUDA out of memory")
        self.global_step += 1
        return {"grpo_loss": 0.0, "optimizer_step_skipped": True, "step": self.global_step}
