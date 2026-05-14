"""
JumpLLaDA Model
================
Pure jump process language model based on Transformer Encoder architecture.
Uses Adaptive Layer Norm (adaLN) for time conditioning, since the jump rate
lambda_t = kappa_dot_t / (1 - kappa_t) is time-dependent.

Architecture: RMSNorm + SwiGLU + RoPE + adaLN (similar to LLaMA/LLaDA but
with time conditioning added).

Target: ~1.4B parameters (matching DCLM 1B scale).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


# ============================================================================
# Building Blocks
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm: modulates scale/shift based on time embedding.

    This is the adaLN mechanism from DiT (Diffusion Transformers).
    Given time embedding c, produces scale (gamma) and shift (beta):
        output = gamma * RMSNorm(x) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # No learnable weight here — scale/shift come from time embedding
        # We keep a base weight for when no conditioning is applied
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input
            scale: (B, 1, D) or (B, D) scale from time embedding
            shift: (B, 1, D) or (B, D) shift from time embedding
        """
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)
        return self._norm(x) * self.weight * (1 + scale) + shift


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE to input tensor x."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    """Multi-head self-attention (bidirectional, no causal mask)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (bidirectional — no causal mask)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(attn)


# ============================================================================
# Transformer Block with adaLN
# ============================================================================

class JumpTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm for time conditioning.

    Structure:
        x = x + Attention(adaLN(x, t))
        x = x + FFN(adaLN(x, t))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = AdaptiveRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn_norm = AdaptiveRMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = Attention(config.hidden_dim, config.n_heads, config.dropout)
        self.ffn = SwiGLU(config.hidden_dim, config.intermediate_dim, config.dropout)

        # adaLN modulation: time_embed -> (scale1, shift1, scale2, shift2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, 4 * config.hidden_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Compute modulation parameters from time embedding
        mod = self.adaLN_modulation(time_emb)  # (B, 4*D)
        scale1, shift1, scale2, shift2 = mod.chunk(4, dim=-1)

        # Self-attention with adaLN
        h = self.attn_norm(x, scale1, shift1)
        x = x + self.attn(h, cos, sin)

        # FFN with adaLN
        h = self.ffn_norm(x, scale2, shift2)
        x = x + self.ffn(h)

        return x


# ============================================================================
# Time Embedding
# ============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal time step embedding followed by MLP.

    Maps scalar time t ∈ [0, 1] to a vector of dimension `embed_dim`.
    """

    def __init__(self, embed_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.frequency_dim = frequency_dim

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal embedding from scalar t."""
        half_dim = self.frequency_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, frequency_dim)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar time steps in [0, 1]
        Returns:
            (B, embed_dim) time embeddings
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = self.sinusoidal_embedding(t)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


# ============================================================================
# JumpLLaDA Model
# ============================================================================

class JumpLLaDA(nn.Module):
    """Pure Jump Process Language Model.

    Transformer Encoder with:
    - Bidirectional self-attention (no causal mask)
    - RoPE positional encoding
    - SwiGLU FFN
    - Adaptive RMSNorm (adaLN) for time conditioning
    - Final linear head → vocab logits

    The model predicts p_theta(x_0 | x_t, t) — the probability of the clean
    token given the noisy input at time t.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Time embedding
        self.time_embedding = TimestepEmbedding(config.time_embed_dim)

        # Optional post-embedding norm
        self.post_embed_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps) if config.post_embed_norm else None

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            config.hidden_dim // config.n_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            JumpTransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # Output head
        if config.weight_tying:
            self.output_head = None  # Will use token_embedding.weight
        else:
            self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs (may contain mask_id tokens)
            t: (B,) time steps in [0, 1]

        Returns:
            logits: (B, L, V) vocabulary logits
        """
        B, L = input_ids.shape

        # Token embedding
        x = self.token_embedding(input_ids)

        if self.post_embed_norm is not None:
            x = self.post_embed_norm(x)

        # Time embedding
        time_emb = self.time_embedding(t)  # (B, time_embed_dim)

        # RoPE
        cos, sin = self.rotary_emb(L)

        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, time_emb, use_reentrant=False
                )
            else:
                x = layer(x, cos, sin, time_emb)

        # Final norm
        x = self.final_norm(x)

        # Output logits
        if self.config.weight_tying:
            logits = F.linear(x, self.token_embedding.weight).clone()
        else:
            logits = self.output_head(x).clone()

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Utility
# ============================================================================

def create_model(config: Optional[ModelConfig] = None) -> JumpLLaDA:
    """Create a JumpLLaDA model with the given config."""
    if config is None:
        config = ModelConfig()
    model = JumpLLaDA(config)
    return model


if __name__ == "__main__":
    # Quick sanity check
    config = ModelConfig()
    model = create_model(config)

    total_params = model.count_parameters()
    print(f"Model config: hidden={config.hidden_dim}, layers={config.n_layers}, "
          f"heads={config.n_heads}, ffn={config.intermediate_dim}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Test forward pass
    B, L = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    t = torch.rand(B)
    logits = model(input_ids, t)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("✓ Forward pass successful!")
