"""
Train SEDD (Score Entropy Discrete Diffusion) model. Run as:

python -m scripts.sedd_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.sedd_train

CPU/MPS smoke test:
python -m scripts.sedd_train --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.sedd import SEDD, SEDDConfig
from nanochat.noise import sample_time, corrupt_absorbing, get_sigma_bar
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.flash_attention import HAS_FA3
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train SEDD diffusion model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# SEDD-specific
parser.add_argument("--noise-schedule", type=str, default="loglinear", choices=["geometric", "loglinear"], help="noise schedule")
parser.add_argument("--sigma-min", type=float, default=1e-4, help="min noise level (geometric schedule)")
parser.add_argument("--sigma-max", type=float, default=20.0, help="max noise level (geometric schedule)")
parser.add_argument("--time-embd-dim", type=int, default=128, help="time embedding dimension")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (-1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto-compute)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for score_head parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for Muon optimizer")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--sample-steps", type=int, default=128, help="number of denoising steps for sampling")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Compute init and wandb

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sedd", name=args.run, config=user_config)

if HAS_FA3:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build SEDD model on meta device (shapes/dtypes only)."""
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = SEDDConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        mask_token_id=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        time_embd_dim=args.time_embd_dim,
        noise_schedule=args.noise_schedule,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
    )
    with torch.device("meta"):
        model_meta = SEDD(config)
    return model_meta

model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

# Checkpoint resume
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"sedd_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "sedd_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# FP8 training initialization

if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        print0(f"FP8 training enabled ({args.fp8_recipe}) - converted {num_fp8}/{num_linear} linear layers")

@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))
    if not fp8_locations:
        yield
        return
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(fp8_module.in_features, fp8_module.out_features, bias=fp8_module.bias is not None, device=fp8_module.weight.device, dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Scaling laws for batch size and LR

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Scaling params for horizon calculation
def get_scaling_params(m):
    pc = m.num_scaling_params()
    return pc['block_matrices'] + pc['score_head']
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

# Batch size
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# LR scaling
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

# Weight decay scaling
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

# -----------------------------------------------------------------------------
# Initialize Optimizer

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=(args.adam_beta1, args.adam_beta2),
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# DataLoaders

dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Training horizon and schedulers

assert args.num_iterations > 0 or args.target_param_data_ratio > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")

# LR schedule
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Evaluation helpers

@torch.no_grad()
def evaluate_val_loss(model, val_loader, eval_steps, config):
    """Compute average score entropy loss on validation data."""
    total_loss = 0.0
    count = 0
    for i, (x, _y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        B = x.size(0)
        t = sample_time(B, x.device)
        x_t, _is_masked = corrupt_absorbing(x, t, config.mask_token_id, config.noise_schedule, config.sigma_min, config.sigma_max)
        loss = model(x_t, t, x)
        total_loss += loss.item()
        count += 1
    if ddp and count > 0:
        import torch.distributed as dist
        loss_tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor)
        return (loss_tensor[0] / loss_tensor[1]).item()
    return total_loss / max(count, 1)


@torch.no_grad()
def sample_absorbing(model, config, tokenizer, num_steps=128, temperature=1.0, seq_len=256):
    """
    Generate text via iterative unmasking (tau-leaping) for absorbing diffusion.

    This is the correct sampler for absorbing noise processes (matching the
    official SEDD repo's `_sample_absorb`). At each step:
    1. Get scores from model
    2. With probability 1 - exp(-dsigma), decide to unmask each MASK position
    3. For positions being unmasked, sample tokens from the score distribution
    4. Final step: force-unmask all remaining MASK positions

    The Analytic Predictor (Eq. 19) is designed for the uniform noise process
    and degenerates for absorbing (transp_transition gives all mass on current
    token, so MASK positions can never unmask).
    """
    dev = model.get_device()
    B, T = 1, seq_len
    mask_id = config.mask_token_id

    # Start from all MASK tokens
    x = torch.full((B, T), mask_id, dtype=torch.long, device=dev)

    # Time discretization: go from t=1 to t=eps
    eps = 1e-3
    ts = torch.linspace(1, eps, num_steps + 1)

    for i in range(num_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), t_now.item(), device=dev)

        # Get scores from model
        log_scores = model(x, t_batch)  # (B, T, vocab_size + 1)
        log_scores = log_scores.float()
        if temperature != 1.0:
            log_scores = log_scores / temperature
        score = log_scores.exp()

        # Sigma decrement for this step
        sigma_now = get_sigma_bar(t_batch, config.noise_schedule, config.sigma_min, config.sigma_max)
        sigma_next = get_sigma_bar(torch.full((B,), t_next.item(), device=dev), config.noise_schedule, config.sigma_min, config.sigma_max)
        dsigma = sigma_now - sigma_next  # (B,) positive

        # Probability of unmasking each position (same rate as forward masking)
        move_chance = 1 - torch.exp(-dsigma)  # (B,)

        # Decide which positions to unmask: only currently-masked positions
        move = torch.rand(B, T, device=dev) < move_chance[:, None]
        move = move & (x == mask_id)

        # For positions to unmask, sample from score distribution (exclude mask dim)
        token_probs = score[..., :-1].clamp(min=0)  # (B, T, vocab_size)
        token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        new_tokens = torch.multinomial(token_probs.view(-1, config.vocab_size), 1).squeeze(-1).view(B, T)

        x = torch.where(move, new_tokens, x)

    # Final denoising step: unmask ALL remaining masked positions
    t_batch = torch.full((B,), eps, device=dev)
    log_scores = model(x, t_batch).float()
    if temperature != 1.0:
        log_scores = log_scores / temperature
    score = log_scores.exp()

    token_probs = score[..., :-1].clamp(min=0)  # (B, T, vocab_size)
    # Fallback to uniform for zero-prob rows (can happen early in training)
    zero_rows = token_probs.sum(dim=-1, keepdim=True) < 1e-8
    uniform = torch.ones_like(token_probs) / config.vocab_size
    token_probs = torch.where(zero_rows.expand_as(token_probs), uniform, token_probs)
    token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True)
    new_tokens = torch.multinomial(token_probs.view(-1, config.vocab_size), 1).squeeze(-1).view(B, T)

    still_masked = (x == mask_id)
    x = torch.where(still_masked, new_tokens, x)

    return tokenizer.decode(x[0].tolist())


# -----------------------------------------------------------------------------
# Training loop

if not resuming:
    step = 0
    val_loss = None
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data.get("val_loss")
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluate val loss
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_loss = evaluate_val_loss(model, val_loader, eval_steps, model_config)
        print0(f"Step {step:05d} | Validation score entropy loss: {val_loss:.6f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({"step": step, "total_training_time": total_training_time, "val/loss": val_loss})
        model.train()

    # Sample from model
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            sample_text = sample_absorbing(orig_model, model_config, tokenizer, num_steps=args.sample_steps, seq_len=min(256, args.max_seq_len))
        print0(f"Step {step:05d} | Sample: {sample_text[:200]}")
        wandb_run.log({"step": step, "sample": wandb.Html(f"<pre>{sample_text[:500]}</pre>") if not use_dummy_wandb else None})
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_loss": val_loss,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # x is the clean token sequence, y is ignored (shifted targets for AR)
        x0 = x
        B = x0.size(0)
        t = sample_time(B, x0.device)
        x_t, _is_masked = corrupt_absorbing(x0, t, model_config.mask_token_id, model_config.noise_schedule, model_config.sigma_min, model_config.sigma_max)
        with autocast_ctx:
            loss = model(x_t, t, x0)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_str = f" | eta: {remaining_steps * avg_time_per_step / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })

    # State update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # GC management (same as base_train)
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_loss is not None:
    print0(f"Minimum validation loss: {min_val_loss:.6f}")

wandb_run.finish()
compute_cleanup()
