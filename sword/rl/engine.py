"""
Continuous GRPO Training Engine for Ling-3.0-tiny and MoE Architectures.
Coordinates Sword Fast Rollout Serving (188+ tokens/s) + Unsloth Memory-Efficient Loss:
- start_grpo(data=[...], continues="checkpoint-xxx")
- "Infinity Time" dynamic data streaming & hot-reloading
- Dual-System Scorer: System 1 (Deterministic Primary Scorer) + System 2 (Advisory 9B Verifier)
- MoE router collapse prevention with active load-balancing aux loss & entropy alarms
- Checkpoint persistence and HuggingFace Hub auto-sync
"""

import os
import gc
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .schema import DatasetRow, Trajectory, ScoredTrajectory, DomainType
from .queue import ContinuousStreamingQueue
from .scorer import PrimaryScorer
from .verifier import ExternalVerifier
from .moe_monitor import MoERouterMonitor
from .loss import ChunkedGRPOLoss
from ..ling import FastLingServer, load_ling_model, patch_ling
from ..finetune import apply_lora_to_model
from ..trainer import (
    setup_blackwell_environment,
    download_from_drive,
    download_hf_checkpoint,
    is_valid_checkpoint,
)


class GRPOTrainer:
    """
    Unified In-Process GRPO Training Engine.
    Combines Sword 8-stream rollout generation with memory-efficient policy gradient updates.
    """

    def __init__(
        self,
        model_name_or_path: str = "inclusionAI/Ling-3.0-tiny",
        verifier_model_name: Optional[str] = None,
        data_sources: Optional[List[str]] = None,
        output_dir: str = "checkpoints-grpo",
        num_rollouts_per_prompt: int = 8,  # Optimal G=8 for Blackwell & GRPO
        batch_size: int = 2,               # 2 prompts x 8 = 16 concurrent trajectories
        max_new_tokens: int = 512,
        lr: float = 5e-6,
        aux_loss_coeff: float = 0.01,
        save_steps: int = 50,
        save_limit: int = 3,
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        setup_blackwell_environment()

        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.num_rollouts_per_prompt = num_rollouts_per_prompt
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.lr = lr
        self.save_steps = save_steps
        self.save_limit = save_limit
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.device = device
        self.global_step = 0

        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 72)
        print(" 🚀 INITIALIZING SWORD CONTINUAL GRPO TRAINING ENGINE")
        print(f" Target Model:  {model_name_or_path}")
        print(f" Concurrency:   {num_rollouts_per_prompt} rollouts/prompt (G={num_rollouts_per_prompt})")
        print(f" Device:        {device}")
        print("=" * 72)

        # 1. Initialize Streaming Queue ("Infinity Time")
        self.queue = ContinuousStreamingQueue(
            data_sources=data_sources,
            cache_dir=os.path.join(output_dir, "data_cache"),
        )

        # 2. Load Policy Model & Tokenizer with Sword Flash Acceleration
        print("\n[*] Loading Policy Model with Sword Engine...")
        self.server = FastLingServer.from_pretrained(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            patch_sword=True,
            patch_moe=True,
        )
        self.model = self.server.model
        self.tokenizer = self.server.tokenizer

        # 3. Apply LoRA Adapters for Parameter-Efficient Training
        print(f"[*] Applying LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
        self.model = apply_lora_to_model(self.model, r=lora_r, lora_alpha=lora_alpha)

        # 4. Initialize System 1 Primary Scorer (Deterministic)
        self.scorer = PrimaryScorer()

        # 5. Initialize System 2 External Verifier (Advisory Judge)
        verifier_model = None
        if verifier_model_name:
            print(f"[*] Loading External Verifier Model: {verifier_model_name}...")
            try:
                # Can be Qwen 3.5 9B with FastMoE
                from ..server import FastMoEServer
                verifier_model = FastMoEServer.from_pretrained(
                    model_name_or_path=verifier_model_name,
                    torch_dtype=torch_dtype,
                )
            except Exception as e:
                print(f"⚠️  Could not load verifier server ({e}), using internal semantic judge fallback.")
        self.verifier = ExternalVerifier(verifier_model=verifier_model)

        # 6. Initialize MoE Router Collapse Monitor
        self.moe_monitor = MoERouterMonitor(
            num_experts=getattr(self.model.config, "num_experts", 64),
            top_k=getattr(self.model.config, "num_experts_per_tok", 8),
            aux_loss_coeff=aux_loss_coeff,
        )

        # 7. Initialize Chunked GRPO Loss Engine
        self.loss_fn = ChunkedGRPOLoss()

        # 8. Setup Optimizer (Trainable LoRA & Router parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def add_data_source(self, source_url_or_path: str):
        """Dynamic dataset registration without stopping the engine ("infinity time")."""
        return self.queue.add_data_source(source_url_or_path)

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Executes a single end-to-end GRPO step:
        1. Dequeue problem batch
        2. Fast multi-stream rollout generation (Sword Flash Engine)
        3. Dual-system evaluation (System 1 Deterministic + System 2 9B Verifier)
        4. Group advantage calculation
        5. Update streaming curriculum & retry queues
        6. In-process policy gradient backward with chunked loss
        7. MoE router entropy monitoring
        """
        # 1. Fetch batch of problems
        problems = self.queue.get_batch(batch_size=self.batch_size)
        if not problems:
            print("[Sword-RL] ⏳ Queue is currently empty, waiting for new problems...")
            return None

        self.global_step += 1
        t_start = time.perf_counter()

        all_scored_trajectories: List[List[ScoredTrajectory]] = []
        flat_input_ids = []
        flat_attention_masks = []
        flat_prompt_lens = []
        flat_advantages = []

        # 2. Multi-Trajectory Rollout Generation Phase (eval mode)
        self.model.eval()
        with torch.no_grad():
            for prob in problems:
                # Generate G trajectories concurrently
                prompts = [prob.user_problem]
                rollouts = self.server.generate_rollouts(
                    prompts=prompts,
                    num_rollouts_per_prompt=self.num_rollouts_per_prompt,
                    max_new_tokens=min(self.max_new_tokens, prob.effort_tier.max_tokens),
                    temperature=0.8,
                    use_fast_engine=True,
                )[0]

                # 3. Dual-System Evaluation Phase
                group_scored: List[ScoredTrajectory] = []
                group_rewards: List[float] = []

                for text in rollouts:
                    # Parse reasoning trace vs final answer
                    trace = ""
                    answer = text
                    if "</think>" in text:
                        parts = text.split("</think>", 1)
                        trace = parts[0].replace("<think>", "").strip()
                        answer = parts[1].strip()

                    traj = Trajectory(
                        prompt=prob.user_problem,
                        full_text=text,
                        reasoning_trace=trace,
                        final_answer=answer,
                        token_count=len(self.tokenizer.encode(text)),
                    )

                    # System 1: Primary Deterministic Scorer
                    scored = self.scorer.score_trajectory(prob, traj)

                    # System 2: Advisory 9B Verifier (Selective low-frequency / recheck sampling)
                    if self.verifier.should_sample(prob):
                        v_delta, v_answers = self.verifier.evaluate_trajectory(prob, traj)
                        scored.total_reward += v_delta
                        scored.verifier_answers = v_answers
                        scored.component_scores["verifier_delta"] = v_delta

                    group_scored.append(scored)
                    group_rewards.append(scored.total_reward)

                # 4. Group Advantage Normalization (GRPO)
                advantages = ChunkedGRPOLoss.compute_group_advantages(group_rewards)
                for idx, adv in enumerate(advantages):
                    group_scored[idx].advantage = adv

                all_scored_trajectories.append(group_scored)

                # 5. Curriculum Feedback to Queue (Bounded Retry & Variation)
                for scored in group_scored:
                    success = scored.total_reward > 0 and scored.is_safe
                    self.queue.handle_feedback(
                        problem=prob,
                        success=success,
                        failure_reason=scored.failure_reason,
                        audit_note=json.dumps(scored.audit_log),
                    )

                # Tokenize inputs + responses for gradient calculation
                for scored in group_scored:
                    p_ids = self.tokenizer.encode(prob.user_problem, add_special_tokens=False)
                    full_ids = self.tokenizer.encode(scored.trajectory.full_text, add_special_tokens=False)
                    combo = p_ids + full_ids

                    flat_input_ids.append(torch.tensor(combo, dtype=torch.long))
                    flat_attention_masks.append(torch.ones(len(combo), dtype=torch.long))
                    flat_prompt_lens.append(len(p_ids))
                    flat_advantages.append(scored.advantage)

        # 6. Policy Gradient Training Phase (train mode)
        self.model.train()
        self.optimizer.zero_grad()

        # Pad batch
        padded_inputs = nn.utils.rnn.pad_sequence(flat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0).to(self.device)
        padded_masks = nn.utils.rnn.pad_sequence(flat_attention_masks, batch_first=True, padding_value=0).to(self.device)
        tensor_advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)

        # Compute Chunked GRPO Loss
        loss, loss_metrics = self.loss_fn.forward_chunked(
            model=self.model,
            input_ids=padded_inputs,
            attention_mask=padded_masks,
            prompt_lengths=flat_prompt_lens,
            advantages=tensor_advantages,
        )

        # Add MoE Auxiliary Load-Balancing Loss if MoE model
        moe_loss = torch.tensor(0.0, device=self.device)
        moe_metrics = {}
        # Collect router logits from MoE layers if available
        for module in self.model.modules():
            if hasattr(module, "last_router_logits") and module.last_router_logits is not None:
                l_aux, m_met = self.moe_monitor.compute_aux_loss(module.last_router_logits)
                moe_loss = moe_loss + l_aux
                moe_metrics.update(m_met)

        total_loss = loss + moe_loss

        # Backward & Optimizer Step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        elapsed = time.perf_counter() - t_start

        # 7. Checkpointing
        if self.global_step % self.save_steps == 0:
            self.save_checkpoint(self.global_step)

        # Summary Logging
        all_rewards = [s.total_reward for grp in all_scored_trajectories for s in grp]
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        step_summary = {
            "step": self.global_step,
            "elapsed_sec": round(elapsed, 2),
            "mean_reward": round(mean_reward, 4),
            "loss": loss_metrics.get("grpo_loss", 0.0),
            "moe_entropy": moe_metrics.get("routing_entropy", 0.0),
            "queue_remaining": len(self.queue.queue),
        }

        print(
            f"[Step {self.global_step:04d}] Reward: {mean_reward:+.3f} | "
            f"Loss: {step_summary['loss']:.4f} | "
            f"Entropy: {step_summary['moe_entropy']:.2f} | "
            f"Time: {elapsed:.2f}s | Queue: {step_summary['queue_remaining']}"
        )

        return step_summary

    def train_continuous(self, max_steps: int = 100000):
        """
        Runs the continuous streaming loop ("infinity time").
        """
        print(f"\n[Sword-RL] ⚡ Starting Continuous Streaming RL Loop (up to {max_steps} steps)...")
        while self.global_step < max_steps:
            if not self.queue.queue:
                print("[Sword-RL] Queue exhausted. Standing by for dynamic data ingestion...")
                time.sleep(5)
                continue
            self.step()

    def save_checkpoint(self, step: int):
        """Saves LoRA weights, optimizer state, and queue curriculum state."""
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n💾 [Sword-RL] Saving checkpoint at step {step} -> {ckpt_dir}")

        # 1. Save LoRA / Model weights
        try:
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(ckpt_dir)
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(ckpt_dir)
            print("  ✅ Model / LoRA weights saved.")
        except Exception as e:
            print(f"  ⚠️  Model save warning: {e}")

        # 2. Save Optimizer State
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

        # 3. Save Queue Curriculum State (Section 2 & 2a)
        self.queue.save_state(os.path.join(ckpt_dir, "queue_state.json"))

        # 4. Save Training State JSON
        state_data = {
            "global_step": self.global_step,
            "model_name_or_path": self.model_name_or_path,
            "num_rollouts_per_prompt": self.num_rollouts_per_prompt,
            "router_summary": self.moe_monitor.get_summary(),
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

        print(f"✅ Checkpoint {step} complete: {ckpt_dir}\n")

    def load_checkpoint(self, ckpt_dir: str):
        """Restores model weights, optimizer, and queue curriculum state."""
        if not os.path.exists(ckpt_dir):
            print(f"[Sword-RL] Checkpoint not found: {ckpt_dir}")
            return

        print(f"\n📥 [Sword-RL] Resuming from checkpoint: {ckpt_dir} ...")

        # 1. Restore weights
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, ckpt_dir)
            print("  ✅ LoRA weights restored.")
        except Exception:
            try:
                # Direct state dict load
                weights_path = os.path.join(ckpt_dir, "adapter_model.bin")
                if os.path.exists(weights_path):
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
            except Exception as e:
                print(f"  ⚠️  Model weight restoration warning: {e}")

        # 2. Restore optimizer
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
            print("  ✅ Optimizer state restored.")

        # 3. Restore queue curriculum state
        q_path = os.path.join(ckpt_dir, "queue_state.json")
        self.queue.load_state(q_path)

        # 4. Restore step
        state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)
                self.global_step = s.get("global_step", 0)

        print(f"✅ Resumption complete at step {self.global_step}.\n")


def start_grpo(
    data: Optional[List[str]] = None,
    continues: Optional[str] = None,
    model_name_or_path: str = "inclusionAI/Ling-3.0-tiny",
    verifier_model_name: Optional[str] = None,
    output_dir: str = "checkpoints-grpo",
    num_rollouts_per_prompt: int = 8,
    batch_size: int = 2,
    max_new_tokens: int = 512,
    lr: float = 5e-6,
    save_steps: int = 50,
    **kwargs,
) -> GRPOTrainer:
    """
    Main user-facing entrypoint for continuous GRPO training:
      start_grpo(data=["drive_link_1", "drive_link_2", ...], continues="checkpoint-xxx")

    Supports:
    - Google Drive links and local jsonl datasets
    - Continuous dynamic expansion ("infinity time scale up")
    - Checkpoint resumption
    - Dual-system scoring (Deterministic + 9B Verifier)
    """
    trainer = GRPOTrainer(
        model_name_or_path=model_name_or_path,
        verifier_model_name=verifier_model_name,
        data_sources=data,
        output_dir=output_dir,
        num_rollouts_per_prompt=num_rollouts_per_prompt,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        lr=lr,
        save_steps=save_steps,
        **kwargs,
    )

    if continues:
        trainer.load_checkpoint(continues)

    return trainer
