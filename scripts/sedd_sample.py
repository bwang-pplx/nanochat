"""
Sample from a trained SEDD model using iterative unmasking (tau-leaping).

Usage:
    python -m scripts.sedd_sample --checkpoint-dir <path> --step <N>
    python -m scripts.sedd_sample --checkpoint-dir <path> --step <N> --num-samples 5
    python -m scripts.sedd_sample --checkpoint-dir <path> --step <N> --prompt "The capital of France"
"""

import os
import argparse
from contextlib import nullcontext

import torch

from nanochat.sedd import SEDD, SEDDConfig
from nanochat.noise import get_sigma_bar
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import autodetect_device_type, get_base_dir

parser = argparse.ArgumentParser(description="Sample from trained SEDD model")
parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint directory (default: auto from depth)")
parser.add_argument("--depth", type=int, default=20, help="model depth (used for auto checkpoint dir)")
parser.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--num-samples", type=int, default=3, help="number of samples to generate")
parser.add_argument("--seq-len", type=int, default=256, help="sequence length to generate")
parser.add_argument("--num-steps", type=int, default=128, help="number of denoising steps")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--prompt", type=str, default=None, help="optional prompt for infilling (placed at start)")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args = parser.parse_args()

# Device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
if device_type == "cuda":
    torch.cuda.set_device(device)

# Tokenizer
tokenizer = get_tokenizer()

# Load checkpoint
base_dir = get_base_dir()
if args.checkpoint_dir is None:
    checkpoint_dir = os.path.join(base_dir, "sedd_checkpoints", f"sedd_d{args.depth}")
else:
    checkpoint_dir = args.checkpoint_dir

print(f"Loading checkpoint from {checkpoint_dir}")
model_data, _, meta_data = load_checkpoint(checkpoint_dir, args.step, device, load_optimizer=False)
config_dict = meta_data["model_config"]
config = SEDDConfig(**config_dict)

# Build and load model
with torch.device("meta"):
    model = SEDD(config)
model.to_empty(device=device)
model.load_state_dict(model_data, strict=True, assign=True)
del model_data
model.eval()

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()


@torch.no_grad()
def sample_absorbing(model, config, seq_len, num_steps, temperature, prompt_tokens=None, seed=42):
    """
    Generate via iterative unmasking (tau-leaping) for absorbing diffusion.

    This is the correct sampler for absorbing noise processes (matching the
    official SEDD repo). At each step:
    1. Get scores from model
    2. With probability 1 - exp(-dsigma), decide to unmask each MASK position
    3. For positions being unmasked, sample tokens from the score distribution
    4. Final step: force-unmask all remaining MASK positions

    If prompt_tokens is provided, those tokens are fixed at the start of the
    sequence and never changed during generation (infilling mode).
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    B, T = 1, seq_len
    mask_id = config.mask_token_id

    # Initialize with all MASK tokens
    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)

    # If prompt provided, fix those tokens
    fixed_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    if prompt_tokens is not None:
        prompt_len = min(len(prompt_tokens), T)
        x[0, :prompt_len] = torch.tensor(prompt_tokens[:prompt_len], device=device)
        fixed_mask[0, :prompt_len] = True

    # Time discretization: go from t=1 to t=eps
    eps = 1e-3
    ts = torch.linspace(1, eps, num_steps + 1)

    for i in range(num_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), t_now.item(), device=device)

        with autocast_ctx:
            log_scores = model(x, t_batch)
        log_scores = log_scores.float()
        if temperature != 1.0:
            log_scores = log_scores / temperature
        score = log_scores.exp()

        # Sigma decrement for this step
        sigma_now = get_sigma_bar(t_batch, config.noise_schedule, config.sigma_min, config.sigma_max)
        sigma_next = get_sigma_bar(torch.full((B,), t_next.item(), device=device), config.noise_schedule, config.sigma_min, config.sigma_max)
        dsigma = sigma_now - sigma_next  # (B,) positive

        # Probability of unmasking each position
        move_chance = 1 - torch.exp(-dsigma)  # (B,)

        # Decide which positions to unmask: only currently-masked positions
        move = torch.rand(B, T, device=device, generator=rng) < move_chance[:, None]
        move = move & (x == mask_id) & ~fixed_mask

        # For positions to unmask, sample from score distribution (exclude mask dim)
        token_probs = score[..., :-1].clamp(min=0)  # (B, T, vocab_size)
        token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        new_tokens = torch.multinomial(token_probs.view(-1, config.vocab_size), 1, generator=rng).squeeze(-1).view(B, T)

        x = torch.where(move, new_tokens, x)

    # Final denoising step: unmask ALL remaining masked positions
    t_batch = torch.full((B,), eps, device=device)
    with autocast_ctx:
        log_scores = model(x, t_batch)
    log_scores = log_scores.float()
    if temperature != 1.0:
        log_scores = log_scores / temperature
    score = log_scores.exp()

    token_probs = score[..., :-1].clamp(min=0)  # (B, T, vocab_size)
    # Fallback to uniform for zero-prob rows (can happen early in training)
    zero_rows = token_probs.sum(dim=-1, keepdim=True) < 1e-8
    uniform = torch.ones_like(token_probs) / config.vocab_size
    token_probs = torch.where(zero_rows.expand_as(token_probs), uniform, token_probs)
    token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True)
    new_tokens = torch.multinomial(token_probs.view(-1, config.vocab_size), 1, generator=rng).squeeze(-1).view(B, T)

    still_masked = (x == mask_id) & ~fixed_mask
    x = torch.where(still_masked, new_tokens, x)

    return x[0].tolist()


# Generate samples
print(f"\nGenerating {args.num_samples} samples (seq_len={args.seq_len}, steps={args.num_steps}, temp={args.temperature}):\n")

prompt_tokens = None
if args.prompt:
    prompt_tokens = tokenizer(args.prompt, prepend="<|bos|>")
    print(f"Prompt: {args.prompt}\n")

for i in range(args.num_samples):
    tokens = sample_absorbing(model, config, args.seq_len, args.num_steps, args.temperature, prompt_tokens, seed=args.seed + i)
    text = tokenizer.decode(tokens)
    print(f"--- Sample {i+1} ---")
    print(text)
    print()
