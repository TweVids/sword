import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sword.ling import (
    FastRMSNormFunction,
    fast_rmsnorm_forward,
    compute_chunked_cross_entropy_loss,
    make_fast_bailing_moe_infer,
)

class MockRMSNorm(nn.Module):
    def __init__(self, hidden_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MockExpert(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MockMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.experts = nn.ModuleList([MockExpert(hidden_dim) for _ in range(num_experts)])

    def moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        active_exp_ids = (tokens_per_expert > 0).nonzero(as_tuple=True)[0]
        cum = torch.cumsum(tokens_per_expert, dim=0)
        starts = (cum - tokens_per_expert)[active_exp_ids].tolist()
        counts = tokens_per_expert[active_exp_ids].tolist()
        exp_list = active_exp_ids.tolist()

        outputs = []
        for exp_id, s_idx, n_tok in zip(exp_list, starts, counts):
            expert = self.experts[exp_id]
            tokens_for_this_expert = sorted_tokens[s_idx:s_idx + n_tok]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(x.dtype)
        )
        return final_out

class TestLingMemoryOptimizations(unittest.TestCase):
    def test_fast_rmsnorm_forward_backward(self):
        torch.manual_seed(42)
        dim = 128
        stock_norm = MockRMSNorm(dim)
        fast_norm = MockRMSNorm(dim)
        fast_norm.load_state_dict(stock_norm.state_dict())

        x_stock = torch.randn(4, 16, dim, requires_grad=True)
        x_fast = x_stock.detach().clone().requires_grad_(True)

        y_stock = stock_norm(x_stock)
        y_fast = FastRMSNormFunction.apply(x_fast, fast_norm.weight, fast_norm.variance_epsilon)

        self.assertTrue(torch.allclose(y_stock, y_fast, atol=1e-5, rtol=1e-5))

        grad_out = torch.randn_like(y_stock)
        y_stock.backward(grad_out)
        y_fast.backward(grad_out)

        self.assertTrue(torch.allclose(x_stock.grad, x_fast.grad, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(stock_norm.weight.grad, fast_norm.weight.grad, atol=1e-4, rtol=1e-4))

    def test_chunked_cross_entropy_exactness(self):
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 32
        hidden_dim = 64
        vocab_size = 1000

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :5] = -100
        labels[:, 20:25] = -100

        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        stock_logits = lm_head(shift_hidden)
        stock_loss = F.cross_entropy(
            stock_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        chunked_loss = compute_chunked_cross_entropy_loss(
            hidden_states=hidden_states,
            lm_head=lm_head,
            labels=labels,
            chunk_size=8,
            ignore_index=-100,
        )

        self.assertTrue(torch.allclose(stock_loss, chunked_loss, atol=1e-5, rtol=1e-5))

    def test_inplace_moe_infer(self):
        torch.manual_seed(42)
        block = MockMoEBlock(num_experts=8, hidden_dim=64)
        fast_infer = make_fast_bailing_moe_infer(block.moe_infer)

        num_tokens = 40
        topk = 4
        x = torch.randn(num_tokens, 64)
        logits = torch.randn(num_tokens, 8)
        _, topk_ids = torch.topk(logits, topk, dim=-1)
        topk_weight = F.softmax(torch.randn(num_tokens, topk), dim=-1)

        stock_out = block.moe_infer(x, topk_ids, topk_weight)
        fast_out = fast_infer(block, x, topk_ids, topk_weight)

        self.assertTrue(torch.allclose(stock_out, fast_out, atol=1e-5, rtol=1e-5))

if __name__ == '__main__':
    unittest.main()
