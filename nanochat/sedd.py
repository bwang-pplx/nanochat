"""
SEDD: Score Entropy Discrete Diffusion model.

Bidirectional transformer that learns to denoise masked tokens via score
entropy loss. Uses adaLN-Zero for time conditioning.

Reference: Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios
of the Data Distribution", ICML 2024.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn
from nanochat.noise import get_sigma_bar, get_dsigma_dt, absorbing_ratio, stable_expm1


@dataclass
class SEDDConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768       # real vocab (tokenizer size)
    mask_token_id: int = 32768    # = vocab_size, internal MASK token
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    time_embd_dim: int = 128      # sinusoidal time embedding dimension
    noise_schedule: str = 'loglinear'
    sigma_min: float = 1e-4
    sigma_max: float = 20.0


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class TimeEmbedding(nn.Module):
    """Sinusoidal time encoding + 2-layer MLP."""
    def __init__(self, time_embd_dim, out_dim):
        super().__init__()
        self.time_embd_dim = time_embd_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embd_dim, out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, t):
        """
        Args:
            t: (B,) float tensor of noise levels (sigma_bar values).
        Returns:
            (B, out_dim) time embeddings.
        """
        half = self.time_embd_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)  # (B, time_embd_dim)
        return F.silu(self.mlp(emb))


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero: projects time embedding to 6 modulation scalars
    per block (gamma1, beta1, alpha1, gamma2, beta2, alpha2).

    gamma/beta modulate RMSNorm, alpha gates the residual.
    Initialized to zero so blocks start as identity.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Linear(n_embd, 6 * n_embd, bias=False)

    def forward(self, t_emb):
        """
        Args:
            t_emb: (B, n_embd) time embedding.
        Returns:
            6 tensors of shape (B, 1, n_embd): gamma1, beta1, alpha1, gamma2, beta2, alpha2.
        """
        out = self.proj(t_emb).unsqueeze(1)  # (B, 1, 6*n_embd)
        return out.chunk(6, dim=-1)


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Bidirectional attention: causal=False, no sliding window
        y = flash_attn.flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class SEDDBlock(nn.Module):
    """
    Transformer block with adaLN-Zero time conditioning:
        x = x + alpha1 * attn((1 + gamma1) * norm(x) + beta1)
        x = x + alpha2 * mlp((1 + gamma2) * norm(x) + beta2)
    """
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = MLP(config)
        self.adaln = AdaLNZero(config.n_embd)

    def forward(self, x, t_emb, cos_sin):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(t_emb)
        # Attention branch
        h = norm(x)
        h = (1 + gamma1) * h + beta1
        x = x + alpha1 * self.attn(h, cos_sin)
        # MLP branch
        h = norm(x)
        h = (1 + gamma2) * h + beta2
        x = x + alpha2 * self.mlp(h)
        return x


class SEDD(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: This __init__ runs in meta device context.
        All actual data initialization happens in init_weights().
        """
        super().__init__()
        self.config = config
        # effective_vocab = vocab_size + 1 (for MASK), padded for efficiency
        effective_vocab = config.vocab_size + 1
        padded_vocab_size = ((effective_vocab + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size
        if padded_vocab_size != effective_vocab:
            print0(f"Padding effective vocab from {effective_vocab} to {padded_vocab_size} for efficiency")

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.time_emb = TimeEmbedding(config.time_embd_dim, config.n_embd)
        self.blocks = nn.ModuleList([SEDDBlock(config) for _ in range(config.n_layer)])
        # Score head: outputs over vocab_size + 1 (real tokens + MASK), for self-score zeroing
        self.score_head = nn.Linear(config.n_embd, config.vocab_size + 1, bias=False)

        # Rotary embeddings (over-allocated, same pattern as GPT)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding (includes MASK token at index vocab_size)
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

        # Score head: zero-init (matches paper's DDitFinalLayer)
        torch.nn.init.zeros_(self.score_head.weight)

        # Time embedding MLP
        for module in self.time_emb.mlp:
            if isinstance(module, nn.Linear):
                torch.nn.init.uniform_(module.weight, -s, s)

        # Transformer blocks
        for block in self.blocks:
            # Attention
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # adaLN-Zero: zero init so blocks start as identity
            torch.nn.init.zeros_(block.adaln.proj.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embedding to bf16 on CUDA
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.wte.weight.device

    def forward(self, x_t, t, x0=None):
        """
        Forward pass.

        Args:
            x_t: Corrupted token ids, shape (B, T). Contains mask_token_id at masked positions.
            t: Time values, shape (B,). Values in [0, 1].
            x0: Clean token ids, shape (B, T). If provided, computes and returns the loss.

        Returns:
            If x0 is provided: scalar loss (score entropy).
            Otherwise: log-score tensor of shape (B, T, vocab_size + 1).
        """
        B, T = x_t.size()
        config = self.config

        # Rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Compute sigma_bar for each sample
        sigma_bar = get_sigma_bar(t, config.noise_schedule, config.sigma_min, config.sigma_max)  # (B,)

        # Time embedding
        t_emb = self.time_emb(sigma_bar)  # (B, n_embd)

        # Token embedding + norm
        x = self.wte(x_t)
        x = norm(x)

        # Transformer blocks with time conditioning
        for block in self.blocks:
            x = block(x, t_emb, cos_sin)
        x = norm(x)

        # Score head: raw network output -> log-scores
        log_scores = self.score_head(x)  # (B, T, vocab_size + 1)
        log_scores = log_scores.float()  # fp32 for numerical stability

        # scale_by_sigma: reparameterize so network outputs correction relative
        # to natural score scale. log_score = raw - log(exp(sigma)-1) - log(V-1)
        esigm1_log = stable_expm1(sigma_bar).log()  # (B,)
        log_scores = log_scores - esigm1_log[:, None, None] - math.log(config.vocab_size)

        # Zero out self-score: enforce s_theta(x_t)_{x_t} = 1 (log = 0)
        log_scores.scatter_(-1, x_t.unsqueeze(-1).long(), 0.0)

        if x0 is not None:
            return self._score_entropy_loss(log_scores, x_t, x0, sigma_bar, t)
        else:
            return log_scores

    def _score_entropy_loss(self, log_scores, x_t, x0, sigma_bar, t):
        """
        Score entropy loss at MASK positions, weighted by dsigma/dt.

        L = E_t[ dsigma/dt * sum_i ( sum_j s_{i,j} - ratio * log(s_{i,x0_i}) + K(ratio) ) ]

        where ratio = 1 / (exp(sigma) - 1), K(a) = a * (log(a) - 1),
        and the sum is only over masked positions i.

        Args:
            log_scores: (B, T, vocab_size + 1) log-score predictions.
            x_t: (B, T) corrupted tokens.
            x0: (B, T) clean tokens.
            sigma_bar: (B,) noise levels.
            t: (B,) time values (for dsigma/dt computation).

        Returns:
            Scalar loss averaged over batch elements.
        """
        B, T, V = log_scores.size()
        config = self.config

        # Only compute loss at MASK positions
        is_masked = (x_t == config.mask_token_id)  # (B, T)

        if not is_masked.any():
            return torch.tensor(0.0, device=log_scores.device, requires_grad=True)

        # Gather log-scores at masked positions
        masked_log_scores = log_scores[is_masked]  # (M, V)
        masked_x0 = x0[is_masked]  # (M,)

        # Ratio per sample: 1 / (exp(sigma) - 1), with expm1 stability
        ratio = absorbing_ratio(sigma_bar)  # (B,)
        ratio_expanded = ratio.unsqueeze(1).expand(B, T)[is_masked]  # (M,)

        # Positive term: sum of exp(log_score) over non-mask tokens (exclude mask dim)
        # The mask token is at index vocab_size (last dim), exclude it from the sum
        pos_term = masked_log_scores[:, :-1].exp().sum(dim=-1)  # (M,)

        # Negative term: ratio * log_score at the correct clean token
        log_score_correct = masked_log_scores.gather(1, masked_x0.unsqueeze(1)).squeeze(1)  # (M,)
        neg_term = ratio_expanded * log_score_correct

        # Constant term: K(ratio) = ratio * (log(ratio) - 1), makes loss non-negative
        const = ratio_expanded * (ratio_expanded.log() - 1)

        # Per-position loss
        loss_per_pos = pos_term - neg_term + const  # (M,)

        # Aggregate per-sample: sum over masked positions within each sample
        # then weight by dsigma/dt for correct time-integral Monte Carlo estimate
        loss_per_sample = torch.zeros(B, device=log_scores.device)
        sample_idx = torch.arange(B, device=log_scores.device).unsqueeze(1).expand(B, T)[is_masked]
        loss_per_sample.scatter_add_(0, sample_idx, loss_per_pos)

        # Weight by dsigma/dt
        dsigma_dt = get_dsigma_dt(t, config.noise_schedule, config.sigma_min, config.sigma_max)  # (B,)
        weighted_loss = dsigma_dt * loss_per_sample

        return weighted_loss.mean()

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        """Set up Muon/AdamW optimizer following GPT pattern."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters into groups
        # AdamW: embeddings, score_head, time_emb, adaLN
        # Muon: attention + MLP matrices (inside blocks, excluding adaLN)
        embedding_params = list(self.wte.parameters())
        score_head_params = list(self.score_head.parameters())
        time_emb_params = list(self.time_emb.parameters())
        adaln_params = []
        matrix_params = []
        for block in self.blocks:
            adaln_params.extend(list(block.adaln.parameters()))
            matrix_params.extend(list(block.attn.parameters()))
            matrix_params.extend(list(block.mlp.parameters()))

        all_params = embedding_params + score_head_params + time_emb_params + adaln_params + matrix_params
        assert len(all_params) == len(list(self.parameters())), \
            f"Parameter count mismatch: {len(all_params)} vs {len(list(self.parameters()))}"

        # Scale LR by 1/sqrt(d_model) (tuned for 768)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=score_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=time_emb_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=adaln_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def num_scaling_params(self):
        """Return parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.wte.parameters())
        score_head = sum(p.numel() for p in self.score_head.parameters())
        time_emb = sum(p.numel() for p in self.time_emb.parameters())
        adaln = sum(p.numel() for p in [b.adaln.proj for b in self.blocks] for p in p.parameters())
        block_matrices = 0
        for block in self.blocks:
            block_matrices += sum(p.numel() for p in block.attn.parameters())
            block_matrices += sum(p.numel() for p in block.mlp.parameters())
        total = wte + score_head + time_emb + adaln + block_matrices
        assert total == sum(p.numel() for p in self.parameters())
        return {
            'wte': wte,
            'score_head': score_head,
            'time_emb': time_emb,
            'adaln': adaln,
            'block_matrices': block_matrices,
            'total': total,
        }

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.wte.weight.numel()  # embedding is just a lookup
        h, q = self.config.n_head, self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # Bidirectional attention: each token attends to all T tokens
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops
