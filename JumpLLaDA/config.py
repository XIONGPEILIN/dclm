"""
JumpLLaDA Configuration
========================
Model and training configuration for the pure jump process language model.
Architecture matches DCLM 1B scale (~1.4B params) with added time conditioning.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer Encoder architecture config (~1.4B params)."""
    hidden_dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    intermediate_dim: int = 5504  # SwiGLU: typically ~2.67x hidden_dim
    vocab_size: int = 50432       # GPT-NeoX tokenizer (DCLM default)
    max_seq_len: int = 2048
    mask_id: int = 50431          # Last token in vocab used as [MASK]
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    time_embed_dim: int = 256     # Dimension for time step embedding (adaLN)
    weight_tying: bool = False
    post_embed_norm: bool = False
    gradient_checkpointing: bool = False


@dataclass
class JumpConfig:
    """Jump process hyperparameters."""
    eps: float = 1e-3              # Noise floor for kappa schedule
    kappa_schedule: str = "linear" # "linear": kappa_t = (1-eps)*t + eps


@dataclass
class TrainingConfig:
    """Training hyperparameters (aligned with DCLM 1B config)."""
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.033
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100000
    lr_schedule: str = "cosine"    # "cosine" or "constant"
    min_lr_ratio: float = 0.1      # min_lr = lr * min_lr_ratio

    # Batch
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    seq_len: int = 2048

    # Precision
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "no"

    # Logging & Saving
    log_interval: int = 10
    save_interval: int = 5000
    eval_interval: int = 1000
    output_dir: str = "./outputs/jump_llada"

    # Data
    dataset_name: Optional[str] = None       # HuggingFace dataset name
    dataset_path: Optional[str] = None       # Local path to tokenized data
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    num_workers: int = 4

    # Wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Resume
    resume_from: Optional[str] = None

    # Seed
    seed: int = 42
