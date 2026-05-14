# JumpLLaDA: Pure Jump Process Language Model

A **pure jump process** (CTMC) language model based on the [Generator Matching](https://arxiv.org/abs/2410.20587) framework, built on the LLaDA architecture.

## Key Differences from LLaDA

| | LLaDA | JumpLLaDA |
|---|---|---|
| **Paradigm** | Masked Diffusion (MDLM) | Pure Jump Process (CTMC) |
| **Loss** | `CE / p_mask` | CTMC ELBO: `λ_t · (rate_term + CE)` |
| **Sampling** | Confidence-ranked unmasking | Independent per-token CTMC jumps |
| **Remasking** | Low-confidence tokens re-masked | No remasking — tokens self-correct |
| **Token Freezing** | Unmasked tokens frozen | All tokens can change at any step |
| **Scheduling** | Block length, steps, confidence | Pure probabilistic (step size `h` only) |

## Architecture

- **1.47B parameters** Transformer Encoder
- Bidirectional attention (no causal mask)
- RoPE + SwiGLU + Adaptive RMSNorm (adaLN for time conditioning)
- GPT-NeoX tokenizer (vocab 50,432) — same as DCLM

## Quick Start

### 1. Prepare Data

```bash
# Download and tokenize C4 dataset (30B tokens)
bash prepare_data.sh c4 30B

# Or use other datasets:
bash prepare_data.sh fineweb-edu 10B
bash prepare_data.sh openwebtext 10B
bash prepare_data.sh slimpajama 30B
```

### 2. Train

```bash
# One-click training (auto-detects GPUs)
bash run_train.sh

# Or with custom settings:
MAX_STEPS=200000 LR=3e-4 bash run_train.sh /path/to/data

# Or directly with accelerate:
accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset_path ./data/tokenized/allenai_c4 \
    --max_steps 100000 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing
```

### 3. Generate

```bash
python generate.py \
    --checkpoint outputs/jump_llada/final/model.pt \
    --prompt "The meaning of life is" \
    --gen_length 128 \
    --steps 128 \
    --temperature 0.8
```

## VRAM Requirements

| Config | VRAM per GPU | GPUs | Effective Batch Size | Notes |
|--------|-------------|------|---------------------|-------|
| BF16 + GradCkpt, BS=2 | ~24 GB | 1× A100-40GB | 16 (accum=8) | Tight fit |
| BF16 + GradCkpt, BS=4 | ~35 GB | 1× A100-80GB | 32 (accum=8) | Recommended |
| BF16 + GradCkpt, BS=4 | ~35 GB | 4× A100-80GB | 128 (accum=8) | Fast training |
| BF16 + GradCkpt, BS=2 | ~20 GB | 8× A6000-48GB | 128 (accum=8) | Budget option |
| BF16, no GradCkpt, BS=4 | ~55 GB | 1× A100-80GB | 32 (accum=8) | Faster but more VRAM |

### Detailed VRAM Breakdown (BF16 + Gradient Checkpointing)

```
Model parameters:    1.47B × 2 bytes (BF16)    =  2.9 GB
Optimizer states:    1.47B × 8 bytes (AdamW)    = 11.8 GB  (fp32 copy + m + v)
Gradients:           1.47B × 2 bytes            =  2.9 GB
Activations (BS=4, seq=2048, grad_ckpt):        ≈  8~12 GB
KV cache / buffers:                             ≈  2~4 GB
────────────────────────────────────────────────────────
Total (BS=4, grad_ckpt):                        ≈ 28~34 GB
Total (BS=2, grad_ckpt):                        ≈ 22~26 GB
```

> **Recommendation**: Use **4× A100-80GB** or **8× A6000-48GB** with gradient checkpointing enabled for comfortable training. A single A100-80GB can train with BS=4 + gradient accumulation.

## File Structure

```
JumpLLaDA/
├── config.py           # Model, jump process, and training configs
├── model.py            # JumpLLaDA Transformer with adaLN
├── jump_process.py     # CTMC forward process, ELBO loss, Euler sampling
├── generate.py         # Text generation interface
├── train.py            # Training script (accelerate)
├── prepare_data.py     # Data download & tokenization
├── prepare_data.sh     # One-click data preparation
├── run_train.sh        # One-click training
├── accelerate_config.yaml
├── requirements.txt
└── README.md
```

## Mathematical Background

The model is based on the **mixture path** on discrete token space:

$$p_t(\cdot|z) = \kappa_t \cdot \delta_z + (1-\kappa_t) \cdot \delta_{\text{[MASK]}}$$

The **CTMC jump process** solving the KFE for this path has:
- **Jump rate**: $\lambda_t = \dot{\kappa}_t / (1-\kappa_t) = 1/(1-t)$
- **Jump distribution**: $J_t(x'|x, z) = \delta_z(x')$ (teleport to target)

The model learns the **marginal rate matrix**:
$$Q_t^\theta(x'|x_t) = \lambda_t \cdot p_\theta(x_0 = x'|x_t, t)$$

**Training loss** (CTMC ELBO, paper Appendix D.2):
$$L = \mathbb{E}_{t,z,x_t}\left[\lambda_t \cdot \left(\sum_{x'\neq x_t} p_\theta(x'|x_t) - \log p_\theta(z|x_t)\right)\right]$$

**Sampling** (CTMC Euler): Each token independently jumps with probability $h \cdot \lambda_t \cdot (1 - p_\theta(x_t|x_t))$.

## Citation

```bibtex
@article{holzschuh2024generator,
  title={Generator Matching: Generative modeling with arbitrary Markov processes},
  author={Holzschuh, Peter and Vegetti, Simone and Thuerey, Nils},
  journal={arXiv preprint arXiv:2410.20587},
  year={2024}
}
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and others},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```
