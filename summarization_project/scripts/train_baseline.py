"""Train the from-scratch Transformer summarization baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.dataset import SummaryDataset, create_dataloader
from summarization.models import SummarizationTransformer, TransformerConfig


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    """Linear warmup then inverse-sqrt decay, common for Transformer training."""

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        return math.sqrt(warmup_steps) / math.sqrt(step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_constant_scheduler(optimizer: torch.optim.Optimizer):
    """Keep learning rate constant when scheduler ablation is requested."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def shift_target(batch: dict, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create decoder input and next-token labels from target sequence."""
    target = batch["target"]

    # Decoder nhận BOS..token_n-1, label là token_1..EOS.
    target_input = target[:, :-1]
    target_output = target[:, 1:]

    # Mask cũng phải cắt theo chiều target_input để khớp logits.
    target_mask = batch["target_mask"][:, :, :-1, :-1]

    return target_input, target_output, target_mask


def run_epoch(
    model: SummarizationTransformer,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
    grad_clip: float,
    pad_id: int,
    desc: str,
) -> float:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc=desc):
        source = batch["source"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_input, target_output, target_mask = shift_target(batch, pad_id)
        target_input = target_input.to(device)
        target_output = target_output.to(device)
        target_mask = target_mask.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(source, target_input, source_mask, target_mask)

            # CrossEntropyLoss nhận [N, vocab] và label [N].
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip and grad_clip > 0:
                    # Gradient clipping tránh spike khi article dài.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def save_checkpoint(
    path: Path,
    model: SummarizationTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: TransformerConfig,
    epoch: int,
    best_val_loss: float,
) -> None:
    """Save everything needed to resume training on the server."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.__dict__,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "data" / "cached" / "train_tokenized.pkl")
    parser.add_argument("--valid-cache", type=Path, default=PROJECT_ROOT / "data" / "cached" / "valid_tokenized.pkl")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm-type", choices=["pre", "post"], default="pre")
    parser.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-share-embeddings", action="store_true")
    parser.add_argument("--no-weight-tying", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_dataset = SummaryDataset(args.train_cache)
    valid_dataset = SummaryDataset(args.valid_cache)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        limit=args.limit_train,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.limit_valid,
    )

    config = TransformerConfig(
        vocab_size=train_dataset.vocab_size,
        pad_id=train_dataset.pad_id,
        d_model=args.d_model,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        num_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        norm_type=args.norm_type,
        activation=args.activation,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        use_scheduler=not args.no_scheduler,
        label_smoothing=args.label_smoothing,
        share_embeddings=not args.no_share_embeddings,
        weight_tying=not args.no_weight_tying,
    )

    device = torch.device(args.device)
    model = SummarizationTransformer(config).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=config.pad_id,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        build_scheduler(optimizer, config.warmup_steps)
        if config.use_scheduler
        else build_constant_scheduler(optimizer)
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")

    print("Baseline Transformer")
    print(f"Device: {device}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Train examples: {len(train_loader.dataset):,}")
    print(f"Valid examples: {len(valid_loader.dataset):,}")
    print(f"Label smoothing: {config.label_smoothing}")
    print(f"Shared embeddings: {config.share_embeddings}")
    print(f"Weight tying: {config.weight_tying}")
    print(f"Norm type: {config.norm_type}")
    print(f"Activation: {config.activation}")
    print(f"Scheduler: {config.use_scheduler}")
    print(f"Grad clip: {config.grad_clip}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            config.grad_clip,
            config.pad_id,
            desc=f"Train {epoch}",
        )
        val_loss = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer=None,
            scheduler=None,
            device=device,
            grad_clip=config.grad_clip,
            pad_id=config.pad_id,
            desc=f"Valid {epoch}",
        )

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - start,
        }
        history.append(epoch_info)
        print(json.dumps(epoch_info, ensure_ascii=False, indent=2))

        save_checkpoint(args.save_dir / "latest.pt", model, optimizer, scheduler, config, epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, scheduler, config, epoch, best_val_loss)

        with (args.save_dir / "history.json").open("w", encoding="utf-8") as f:
            # Log ra file để khi train server có số liệu đưa vào báo cáo.
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
