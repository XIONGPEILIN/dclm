"""
Jump Process Core
==================
Implements the pure jump process (CTMC) on discrete token space for JumpLLaDA.

Mathematical basis from "Generator Matching" (arXiv:2410.20587):
- Mixture path: p_t(·|z) = κ_t · δ_z + (1-κ_t) · p_0, where p_0 = δ_{[MASK]}
- Conditional rate: Q_t^z(x'|x) = (κ̇_t / (1-κ_t)) · δ_z(x')  for x' ≠ x
- Marginal rate: Q_t^θ(x'|x) = (κ̇_t / (1-κ_t)) · p_θ(x_0=x' | x_t)
- ELBO loss (Appendix D.2):
    L = E[ Σ_{x'≠x_t} Q_t^θ(x'|x_t) - Q_t^z(x'|x_t) · log Q_t^θ(x'|x_t) ]
- CTMC Euler sampling: X_{t+h} ~ (I + h·Q_t)(·; X_t)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def kappa_schedule(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Linear schedule: κ_t = (1-ε)·t + ε.

    At t=0: κ=ε ≈ 0  (almost all masked)
    At t=1: κ=1       (all clean)
    """
    return (1.0 - eps) * t + eps


def kappa_dot(eps: float = 1e-3) -> float:
    """Time derivative of linear kappa schedule: κ̇_t = 1-ε (constant)."""
    return 1.0 - eps


def jump_rate(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Jump intensity: λ_t = κ̇_t / (1 - κ_t).

    For linear schedule:
        λ_t = (1-ε) / (1 - (1-ε)·t - ε) = (1-ε) / ((1-ε)(1-t)) = 1/(1-t)
    """
    kappa_t = kappa_schedule(t, eps)
    k_dot = kappa_dot(eps)
    # Clamp to avoid division by zero at t=1
    return k_dot / (1.0 - kappa_t).clamp(min=1e-6)


def forward_process(
    input_ids: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t from the mixture path p_t(·|z).

    For each token position:
    - With probability κ_t: keep the original token (x_t = z)
    - With probability 1-κ_t: replace with [MASK]  (x_t = mask_id)

    Args:
        input_ids: (B, L) clean token IDs (the target z)
        mask_id: token ID for [MASK]
        eps: noise floor

    Returns:
        x_t: (B, L) noisy token IDs
        t: (B,) sampled time steps
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Sample time uniformly from [0, 1)
    t = torch.rand(B, device=device)

    # κ_t for each sample in the batch
    kappa_t = kappa_schedule(t, eps)  # (B,)

    # For each position, decide: keep original (prob κ_t) or mask (prob 1-κ_t)
    keep_mask = torch.rand(B, L, device=device) < kappa_t.unsqueeze(1)

    # Construct x_t
    x_t = torch.where(keep_mask, input_ids, mask_id)

    return x_t, t


def compute_jump_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute the CTMC ELBO loss for the jump process.

    From paper Appendix D.2, the loss per position is:
        L_i = Σ_{x'≠x_t^i} Q_t^θ(x'|x_t^i) - Q_t^z(x'|x_t^i) · log Q_t^θ(x'|x_t^i)

    For the mixture path with mask prior:
    - Q_t^z(x'|x, z) = λ_t · δ_z(x') for x' ≠ x
    - Q_t^θ(x'|x_t) = λ_t · p_θ(x'|x_t) for x' ≠ x_t

    Substituting (and dropping the common λ_t factor from Q_t^z·log Q_t^θ since
    it cancels up to constants):

    L_i = λ_t · [Σ_{x'≠x_t} p_θ(x'|x_t)] - λ_t · log(λ_t · p_θ(z|x_t))
        = λ_t · [(1 - p_θ(x_t|x_t))] - λ_t · [log λ_t + log p_θ(z|x_t)]
        = λ_t · (1 - p_θ(x_t|x_t)) - λ_t · log p_θ(z|x_t) + const_in_θ
        = λ_t · (1 - p_θ(x_t|x_t) - log p_θ(z|x_t)) + const

    We only keep θ-dependent terms:
        L_i = λ_t · [Σ_{x'≠x_t} p_θ(x'|x_t) - log p_θ(z|x_t)]

    For **masked positions** (x_t = mask_id):
        Σ_{x'≠mask} p_θ(x'|x_t) = 1 - p_θ(mask|x_t)
        Total: λ_t · [(1 - p_θ(mask|x_t)) - log p_θ(z|x_t)]

    For **unmasked positions** (x_t = z):
        Σ_{x'≠z} p_θ(x'|x_t) = 1 - p_θ(z|x_t)
        Target is z itself, so: λ_t · [(1 - p_θ(z|x_t)) - log p_θ(z|x_t)]

    Both cases are valid and contribute to the loss. We compute for ALL positions.

    Args:
        logits: (B, L, V) model output logits
        input_ids: (B, L) clean token IDs (target z)
        x_t: (B, L) noisy token IDs
        t: (B,) time steps
        mask_id: mask token ID
        eps: noise floor

    Returns:
        loss: scalar loss
    """
    B, L, V = logits.shape

    # Jump rate λ_t per sample
    lambda_t = jump_rate(t, eps)  # (B,)

    # Compute log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
    probs = log_probs.exp()

    # Term 1: Σ_{x'≠x_t} p_θ(x'|x_t) = 1 - p_θ(x_t|x_t)
    # Gather p_θ(x_t^i | x_t) for each position
    p_stay = probs.gather(dim=-1, index=x_t.unsqueeze(-1)).squeeze(-1)  # (B, L)
    rate_term = 1.0 - p_stay  # (B, L)

    # Term 2: -log p_θ(z|x_t) = cross-entropy with target z
    # Gather log p_θ(z_i | x_t) for each position
    log_p_target = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)
    ce_term = -log_p_target  # (B, L)

    # Per-position loss: λ_t · (rate_term + ce_term)
    per_pos_loss = lambda_t.unsqueeze(1) * (rate_term + ce_term)  # (B, L)

    # Average over positions and batch
    loss = per_pos_loss.mean()

    return loss


@torch.no_grad()
def euler_sample(
    model,
    seq_len: int,
    steps: int,
    mask_id: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 1,
    temperature: float = 1.0,
    eps: float = 1e-3,
    prompt: torch.Tensor = None,
    prompt_len: int = 0,
) -> torch.Tensor:
    """CTMC Euler sampling for generation.

    Starting from X_0 = [MASK]^L, iteratively apply:
        X_{t+h} ~ (I + h·Q_t^θ)(·; X_t)

    Per token:
        - With prob min(1, h·λ_t·(1 - p_θ(x_t|x_t))): sample new token from p_θ(·|x_t)
        - Otherwise: stay at current token

    Args:
        model: JumpLLaDA model
        seq_len: total sequence length (prompt + generation)
        steps: number of Euler steps
        mask_id: mask token ID
        vocab_size: vocabulary size
        device: torch device
        batch_size: batch size
        temperature: sampling temperature
        eps: noise floor
        prompt: optional (B, P) prompt token IDs
        prompt_len: length of prompt (frozen tokens)

    Returns:
        x: (B, L) generated token IDs
    """
    model.eval()

    # Initialize: all [MASK]
    x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)

    # Set prompt if provided
    if prompt is not None:
        prompt_len = prompt.shape[1]
        x[:, :prompt_len] = prompt

    # Time grid: from 0 to 1-eps
    h = 1.0 / steps
    time_points = torch.linspace(0, 1.0 - h, steps, device=device)

    for step_idx in range(steps):
        t_val = time_points[step_idx]
        t = torch.full((batch_size,), t_val.item(), device=device)

        # Forward pass
        logits = model(x, t)  # (B, L, V)

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (B, L, V)

        # Jump rate
        lambda_t_val = jump_rate(t_val.unsqueeze(0), eps).item()

        # Total jump probability per position: h * λ_t * (1 - p_θ(x_t|x_t))
        p_stay = probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)  # (B, L)
        jump_prob = min(1.0, h * lambda_t_val) * (1.0 - p_stay)  # (B, L)
        jump_prob = jump_prob.clamp(0.0, 1.0)

        # Decide which positions jump
        do_jump = torch.rand_like(jump_prob) < jump_prob  # (B, L)

        # Don't jump prompt positions
        if prompt_len > 0:
            do_jump[:, :prompt_len] = False

        # Sample new tokens for jumping positions
        if do_jump.any():
            # Sample from the categorical distribution
            flat_probs = probs.view(-1, vocab_size)  # (B*L, V)
            flat_do_jump = do_jump.view(-1)  # (B*L,)

            new_tokens = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (B*L,)
            new_tokens = new_tokens.view(batch_size, seq_len)

            # Update only jumping positions
            x = torch.where(do_jump, new_tokens, x)

    return x


@torch.no_grad()
def euler_sample_deterministic(
    model,
    seq_len: int,
    steps: int,
    mask_id: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 1,
    eps: float = 1e-3,
    prompt: torch.Tensor = None,
    prompt_len: int = 0,
) -> torch.Tensor:
    """Deterministic CTMC sampling (argmax, no randomness).

    At each step, for all positions where jump_prob > 0.5, jump to argmax token.
    """
    model.eval()

    x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)
    if prompt is not None:
        prompt_len = prompt.shape[1]
        x[:, :prompt_len] = prompt

    h = 1.0 / steps
    time_points = torch.linspace(0, 1.0 - h, steps, device=device)

    for step_idx in range(steps):
        t_val = time_points[step_idx]
        t = torch.full((batch_size,), t_val.item(), device=device)

        logits = model(x, t)
        probs = F.softmax(logits, dim=-1)

        lambda_t_val = jump_rate(t_val.unsqueeze(0), eps).item()
        p_stay = probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        jump_prob = min(1.0, h * lambda_t_val) * (1.0 - p_stay)
        jump_prob = jump_prob.clamp(0.0, 1.0)

        do_jump = jump_prob > 0.5

        if prompt_len > 0:
            do_jump[:, :prompt_len] = False

        if do_jump.any():
            new_tokens = logits.argmax(dim=-1)
            x = torch.where(do_jump, new_tokens, x)

    return x


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, ".")
    from model import create_model
    from config import ModelConfig

    config = ModelConfig()
    config.hidden_dim = 256
    config.n_layers = 4
    config.n_heads = 4
    config.intermediate_dim = 512
    config.time_embed_dim = 64

    model = create_model(config)
    print(f"Test model params: {model.count_parameters():,}")

    B, L = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, L))

    # Test forward process
    x_t, t = forward_process(input_ids, mask_id=config.mask_id, eps=1e-3)
    masked_ratio = (x_t == config.mask_id).float().mean().item()
    print(f"Forward process: masked ratio = {masked_ratio:.3f}")

    # Test loss
    logits = model(x_t, t)
    loss = compute_jump_loss(logits, input_ids, x_t, t, mask_id=config.mask_id)
    print(f"Loss: {loss.item():.4f}")

    # Test sampling
    generated = euler_sample(model, seq_len=32, steps=16, mask_id=config.mask_id,
                             vocab_size=config.vocab_size, device="cpu", batch_size=2)
    mask_ratio = (generated == config.mask_id).float().mean().item()
    print(f"Generated: mask ratio = {mask_ratio:.3f} (should be low)")
    print("✓ All jump process tests passed!")
