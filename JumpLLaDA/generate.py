"""
JumpLLaDA Generation Interface
================================
User-facing generation with prompt conditioning.
Prompt tokens are frozen; only completion tokens undergo jump process.
"""

import torch
import torch.nn.functional as F
import argparse
from typing import Optional

from config import ModelConfig
from model import create_model
from jump_process import euler_sample


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    temperature: float = 1.0,
    mask_id: int = 50431,
    vocab_size: int = 50432,
) -> torch.Tensor:
    """Generate text conditioned on a prompt using CTMC Euler sampling.

    Args:
        model: JumpLLaDA model
        prompt: (1, P) or (B, P) prompt token IDs
        gen_length: number of tokens to generate
        steps: number of Euler sampling steps
        temperature: sampling temperature (0 = deterministic)
        mask_id: mask token ID
        vocab_size: vocabulary size

    Returns:
        output: (B, P + gen_length) full sequence including prompt
    """
    device = next(model.parameters()).device
    B = prompt.shape[0]
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length

    # Build initial sequence: [prompt | MASK MASK ... MASK]
    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.to(device)

    # Run Euler sampling
    x = euler_sample(
        model=model,
        seq_len=total_len,
        steps=steps,
        mask_id=mask_id,
        vocab_size=vocab_size,
        device=device,
        batch_size=B,
        temperature=temperature,
        prompt=prompt.to(device),
        prompt_len=prompt_len,
    )

    return x


def main():
    parser = argparse.ArgumentParser(description="JumpLLaDA Generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Text prompt for generation")
    parser.add_argument("--gen_length", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--steps", type=int, default=128, help="Number of Euler sampling steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["model_config"])
    model = create_model(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded: {model.count_parameters():,} params")

    # Tokenize prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name
                                               if hasattr(config, 'tokenizer_name')
                                               else "EleutherAI/gpt-neox-20b")
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_ids.shape[1]}")

    # Generate
    output = generate(
        model=model,
        prompt=prompt_ids,
        gen_length=args.gen_length,
        steps=args.steps,
        temperature=args.temperature,
        mask_id=config.mask_id,
        vocab_size=config.vocab_size,
    )

    # Decode
    generated_text = tokenizer.decode(output[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Generated ({args.steps} steps, temp={args.temperature}):")
    print(f"{'='*60}")
    print(f"{args.prompt}{generated_text}")


if __name__ == "__main__":
    main()
