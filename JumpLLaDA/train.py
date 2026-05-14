"""
JumpLLaDA Training Script
===========================
Training with HuggingFace Accelerate for multi-GPU distributed training.

Pure jump process (CTMC) training on tokenized text data.
Supports: multi-GPU, BF16, gradient checkpointing, wandb logging.

Usage:
    # Single GPU
    python train.py --dataset_path ./data/tokenized

    # Multi-GPU with accelerate
    accelerate launch --config_file accelerate_config.yaml train.py --dataset_path ./data/tokenized
"""

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from accelerate import Accelerator
from accelerate.utils import set_seed

from config import ModelConfig, JumpConfig, TrainingConfig
from model import create_model
from jump_process import forward_process, compute_jump_loss, euler_sample


# ============================================================================
# Dataset
# ============================================================================

class TokenizedDataset(Dataset):
    """Memory-mapped tokenized text dataset.

    Expects a directory containing .bin files of uint16 token IDs,
    or a single .bin file. Reads contiguous chunks of `seq_len` tokens.
    """

    def __init__(self, data_path: str, seq_len: int = 2048):
        super().__init__()
        self.seq_len = seq_len

        data_path = Path(data_path)
        if data_path.is_dir():
            # Concatenate all .bin files
            bin_files = sorted(data_path.glob("*.bin"))
            if not bin_files:
                raise ValueError(f"No .bin files found in {data_path}")
            # Memory-map each file
            self.mmaps = []
            self.lengths = []
            total_tokens = 0
            for f in bin_files:
                mmap = torch.from_numpy(
                    __import__('numpy').memmap(str(f), dtype='uint16', mode='r')
                )
                self.mmaps.append(mmap)
                self.lengths.append(len(mmap))
                total_tokens += len(mmap)
            self.total_tokens = total_tokens
        elif data_path.is_file() and data_path.suffix == '.bin':
            mmap = torch.from_numpy(
                __import__('numpy').memmap(str(data_path), dtype='uint16', mode='r')
            )
            self.mmaps = [mmap]
            self.lengths = [len(mmap)]
            self.total_tokens = len(mmap)
        else:
            raise ValueError(f"Expected directory of .bin files or a .bin file, got: {data_path}")

        # Build cumulative lengths for indexing across files
        self.cum_lengths = []
        cumsum = 0
        for l in self.lengths:
            self.cum_lengths.append(cumsum)
            cumsum += l

        self.n_samples = self.total_tokens // seq_len
        print(f"Dataset: {self.total_tokens:,} tokens, {self.n_samples:,} samples of length {seq_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        # Find which file and offset
        tokens = []
        remaining = self.seq_len
        pos = start

        for i, (cum, length) in enumerate(zip(self.cum_lengths, self.lengths)):
            if pos < cum + length and remaining > 0:
                local_start = max(0, pos - cum)
                local_end = min(length, local_start + remaining)
                chunk = self.mmaps[i][local_start:local_end].long()
                tokens.append(chunk)
                remaining -= len(chunk)
                pos = cum + local_end

        if remaining > 0:
            # Wrap around if needed
            tokens.append(self.mmaps[0][:remaining].long())

        return {"input_ids": torch.cat(tokens)}


class HFTokenizedDataset(Dataset):
    """Dataset from HuggingFace datasets library (pre-tokenized)."""

    def __init__(self, hf_dataset, seq_len: int = 2048, text_column: str = "input_ids"):
        self.dataset = hf_dataset
        self.seq_len = seq_len
        self.text_column = text_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset[idx][self.text_column]
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        # Truncate or pad
        if len(tokens) > self.seq_len:
            start = torch.randint(0, len(tokens) - self.seq_len, (1,)).item()
            tokens = tokens[start:start + self.seq_len]
        elif len(tokens) < self.seq_len:
            # Pad with 0 (will be ignored during loss)
            tokens = F.pad(tokens, (0, self.seq_len - len(tokens)), value=0)
        return {"input_ids": tokens}


class StreamingTextDataset(IterableDataset):
    """Streams text from HF dataset, tokenizes on-the-fly, yields seq_len chunks."""
    def __init__(self, dataset_name: str, subset: str = None, split: str = "train", 
                 tokenizer_name: str = "EleutherAI/gpt-neox-20b", seq_len: int = 2048,
                 text_column: str = "text", seed: int = 42):
        super().__init__()
        from datasets import load_dataset
        from transformers import AutoTokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.text_column = text_column
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import torch.distributed as dist

        # Initialize tokenizer inside worker to avoid multiprocessing issues
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Determine worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Determine DDP info
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Total logical partitions = world_size * num_workers
        total_partitions = world_size * num_workers
        partition_id = rank * num_workers + worker_id

        # Load streaming dataset
        ds_kwargs = {"path": self.dataset_name, "split": self.split, "streaming": True}
        if self.subset:
            ds_kwargs["name"] = self.subset
        
        dataset = load_dataset(**ds_kwargs)
        
        # Shuffle with seed and shard for this specific worker
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)
        dataset = dataset.shard(num_shards=total_partitions, index=partition_id)

        buffer = []
        for example in dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue
            
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(step: int, config: TrainingConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)

    if config.lr_schedule == "constant":
        return config.lr

    # Cosine decay
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progress = min(progress, 1.0)
    min_lr = config.lr * config.min_lr_ratio
    return min_lr + 0.5 * (config.lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    # Build configs
    model_config = ModelConfig()
    jump_config = JumpConfig()
    train_config = TrainingConfig()

    # Override from args
    if args.lr is not None:
        train_config.lr = args.lr
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.per_device_batch_size is not None:
        train_config.per_device_batch_size = args.per_device_batch_size
    if args.gradient_accumulation_steps is not None:
        train_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.output_dir is not None:
        train_config.output_dir = args.output_dir
    if args.dataset_path is not None:
        train_config.dataset_path = args.dataset_path
    if args.dataset_name is not None:
        train_config.dataset_name = args.dataset_name
    if args.wandb_project is not None:
        train_config.wandb_project = args.wandb_project
    if args.resume_from is not None:
        train_config.resume_from = args.resume_from
    if args.seq_len is not None:
        train_config.seq_len = args.seq_len
        model_config.max_seq_len = args.seq_len
    if args.gradient_checkpointing:
        model_config.gradient_checkpointing = True
    if args.seed is not None:
        train_config.seed = args.seed
    if args.save_interval is not None:
        train_config.save_interval = args.save_interval

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with="wandb" if train_config.wandb_project else None,
        project_dir=train_config.output_dir,
    )

    set_seed(train_config.seed)

    if accelerator.is_main_process:
        os.makedirs(train_config.output_dir, exist_ok=True)
        print("=" * 60)
        print("JumpLLaDA Training")
        print("=" * 60)

    # Create model
    model = create_model(model_config)

    if accelerator.is_main_process:
        total_params = model.count_parameters()
        print(f"Model: {total_params:,} parameters ({total_params / 1e9:.2f}B)")
        print(f"Architecture: hidden={model_config.hidden_dim}, layers={model_config.n_layers}, "
              f"heads={model_config.n_heads}, ffn={model_config.intermediate_dim}")
        print(f"Mask ID: {model_config.mask_id}, Vocab: {model_config.vocab_size}")

    # Load dataset
    if getattr(args, "streaming", False):
        if accelerator.is_main_process:
            print(f"Using Streaming On-the-Fly Tokenization for {train_config.dataset_name}")
        dataset = StreamingTextDataset(
            dataset_name=train_config.dataset_name,
            subset=getattr(args, "dataset_subset", None),
            seq_len=train_config.seq_len,
            seed=train_config.seed
        )
    elif train_config.dataset_path:
        dataset = TokenizedDataset(train_config.dataset_path, seq_len=train_config.seq_len)
    elif train_config.dataset_name:
        from datasets import load_dataset
        hf_ds = load_dataset(train_config.dataset_name, split="train")
        dataset = HFTokenizedDataset(hf_ds, seq_len=train_config.seq_len)
    else:
        raise ValueError("Must specify --dataset_path or --dataset_name")

    # DataLoader setup: drop_last=True doesn't work with IterableDataset
    drop_last = not isinstance(dataset, torch.utils.data.IterableDataset)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.per_device_batch_size,
        shuffle=drop_last, # Can't shuffle an IterableDataset this way
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    # Optimizer: Muon for 2D weights + AdamW for 1D params (biases, norms, embeddings)
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
    adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]

    if accelerator.is_main_process:
        muon_count = sum(p.numel() for p in muon_params)
        adamw_count = sum(p.numel() for p in adamw_params)
        print(f"Optimizer: Muon ({muon_count:,} params) + AdamW ({adamw_count:,} params)")

    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=train_config.muon_lr,
        momentum=train_config.muon_momentum,
    )
    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=train_config.lr,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        weight_decay=train_config.weight_decay,
    )

    # Prepare with accelerate
    model, optimizer_muon, optimizer_adamw, dataloader = accelerator.prepare(
        model, optimizer_muon, optimizer_adamw, dataloader
    )

    # Resume from checkpoint
    global_step = 0
    if train_config.resume_from:
        accelerator.print(f"Resuming from {train_config.resume_from}")
        accelerator.load_state(train_config.resume_from)
        # Try to recover global_step
        state_path = Path(train_config.resume_from) / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                global_step = state.get("global_step", 0)
        accelerator.print(f"Resumed at step {global_step}")

    # Wandb init
    if train_config.wandb_project and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=train_config.wandb_project,
            config={
                "model": asdict(model_config),
                "jump": asdict(jump_config),
                "training": asdict(train_config),
            },
            init_kwargs={"wandb": {"name": train_config.wandb_run_name or "jump_llada"}},
        )

    if accelerator.is_main_process:
        print(f"Training config: lr={train_config.lr}, bs={train_config.per_device_batch_size}, "
              f"grad_accum={train_config.gradient_accumulation_steps}, max_steps={train_config.max_steps}")
        if hasattr(dataset, "__len__"):
            print(f"Dataset: {len(dataset):,} samples")
        else:
            print("Dataset: Streaming (infinite)")
        print("=" * 60)

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    running_loss = 0.0

    while global_step < train_config.max_steps:
        # Get batch (loop over dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"]  # (B, L)

        with accelerator.accumulate(model):
            # Fix for CUDAGraphs overwrite issue when using torch.compile
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Forward process: sample x_t from mixture path
            x_t, t = forward_process(input_ids, mask_id=model_config.mask_id, eps=jump_config.eps)

            # Model forward
            logits = model(x_t, t)

            # Compute jump ELBO loss
            # Clone logits to prevent CUDA graph overwrite issue if not fixed by mark_step_begin
            loss = compute_jump_loss(
                logits.clone() if hasattr(torch, "compiler") else logits, 
                input_ids, x_t, t,
                mask_id=model_config.mask_id,
                eps=jump_config.eps,
            )

            # Backward
            accelerator.backward(loss)

            # Gradient clipping
            if train_config.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            # Update LR (scale both optimizers proportionally)
            lr_scale = get_lr(global_step, train_config) / max(train_config.lr, 1e-10)
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = train_config.lr * lr_scale
            for param_group in optimizer_muon.param_groups:
                param_group['lr'] = train_config.muon_lr * lr_scale
            lr = train_config.lr * lr_scale  # for logging

            optimizer_muon.step()
            optimizer_adamw.step()
            optimizer_muon.zero_grad()
            optimizer_adamw.zero_grad()

        running_loss += loss.item()

        # Only count steps when we've actually stepped (after gradient accumulation)
        if accelerator.sync_gradients:
            global_step += 1

            # Logging
            if global_step % train_config.log_interval == 0:
                avg_loss = running_loss / train_config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (train_config.log_interval * train_config.per_device_batch_size
                                  * train_config.seq_len * accelerator.num_processes
                                  * train_config.gradient_accumulation_steps) / elapsed

                if accelerator.is_main_process:
                    print(f"step {global_step:>6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                          f"tok/s {tokens_per_sec:.0f} | "
                          f"elapsed {elapsed:.1f}s")

                    if train_config.wandb_project:
                        accelerator.log({
                            "loss": avg_loss,
                            "lr": lr,
                            "tokens_per_sec": tokens_per_sec,
                            "step": global_step,
                        }, step=global_step)

                running_loss = 0.0
                start_time = time.time()

            # Save checkpoint + run inference
            if global_step % train_config.save_interval == 0:
                save_dir = os.path.join(train_config.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_dir)
                if accelerator.is_main_process:
                    # Save training state
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"global_step": global_step}, f)
                    # Save standalone model checkpoint
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({
                        "model_state_dict": unwrapped.state_dict(),
                        "model_config": asdict(model_config),
                        "global_step": global_step,
                    }, os.path.join(save_dir, "model.pt"))
                    print(f"Saved checkpoint at step {global_step}")

                    # Run inference and log to wandb
                    try:
                        unwrapped.eval()
                        device = next(unwrapped.parameters()).device
                        generated = euler_sample(
                            model=unwrapped,
                            seq_len=128,
                            steps=64,
                            mask_id=model_config.mask_id,
                            vocab_size=model_config.vocab_size,
                            device=device,
                            batch_size=4,
                            temperature=0.8,
                        )
                        # Decode generated tokens
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)
                        samples = []
                        for i in range(generated.shape[0]):
                            text = tokenizer.decode(generated[i], skip_special_tokens=True)
                            mask_count = (generated[i] == model_config.mask_id).sum().item()
                            samples.append({
                                "sample_id": i,
                                "text": text[:500],
                                "remaining_masks": mask_count,
                                "step": global_step,
                            })
                            print(f"  Sample {i}: [{mask_count} masks] {text[:200]}")

                        if train_config.wandb_project:
                            import wandb
                            table = wandb.Table(
                                columns=["step", "sample_id", "text", "remaining_masks"],
                                data=[[s["step"], s["sample_id"], s["text"], s["remaining_masks"]] for s in samples]
                            )
                            accelerator.log({"generations": table}, step=global_step)

                        unwrapped.train()
                    except Exception as e:
                        print(f"  Inference failed: {e}")
                        unwrapped.train()

    # Final save
    if accelerator.is_main_process:
        save_dir = os.path.join(train_config.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save({
            "model_state_dict": unwrapped.state_dict(),
            "model_config": asdict(model_config),
            "global_step": global_step,
        }, os.path.join(save_dir, "model.pt"))
        print(f"Training complete! Final model saved to {save_dir}/model.pt")

    if train_config.wandb_project:
        accelerator.end_training()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="JumpLLaDA Training")

    # Data
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to tokenized data directory (.bin files)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_subset", type=str, default=None,
                        help="HuggingFace dataset subset")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream and tokenize on-the-fly directly from HF")

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
