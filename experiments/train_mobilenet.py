"""
Fine-tunes ImageNet MobileNetV2 on EuroSAT RGB as the campaign's second architecture.

Training uses every chip EXCEPT the calibration/evaluation chips of the listed campaign split
configs (seed 0, as in run_campaign.py), so campaign accuracy is measured on unseen chips.
Preprocessing matches the ResNet-18 checkpoint (EuroSAT mean/std, bicubic 224 px).
The recipe is fixed up front (no tuning against the evaluation chips).

Writes checkpoints/mobilenet_v2_eurosat.safetensors and a .json sidecar (recipe, held-out chips,
accuracy on the held-out evaluation chips).

Example (GPU, ~10 min on a T4):
    python experiments/train_mobilenet.py --epochs 3 --device cuda
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.models as models
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.config import DEFAULT_EUROSAT_DATA_DIR, EUROSAT_NUM_CLASSES  # noqa: E402
from aegis.data.eurosat import (  # noqa: E402
    held_out_paths,
    load_eurosat_splits,
    load_training_chips,
    preprocess_chips,
)
from aegis.models.factory import DEFAULT_MOBILENET_CHECKPOINT  # noqa: E402


def parse_split(text: str):
    calib, evals = text.split(":")
    return int(calib), int(evals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--exclude-splits", type=parse_split, nargs="+",
                   default=[(10, 10), (10, 15), (10, 20), (10, 30), (4, 20)],
                   help="calib:eval per-class split configs whose chips are held out of training")
    p.add_argument("--no-imagenet", action="store_true", help="Random init (offline smoke tests only)")
    p.add_argument("--max-train", type=int, default=0, help="Cap training chips (smoke tests only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=DEFAULT_EUROSAT_DATA_DIR)
    p.add_argument("--out", default=DEFAULT_MOBILENET_CHECKPOINT)
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(0)

    held_out = held_out_paths(args.exclude_splits, args.data_dir)
    chips, labels = load_training_chips(held_out, args.data_dir)
    if args.max_train:
        keep = torch.randperm(labels.numel())[:args.max_train]
        chips, labels = chips[keep], labels[keep]
    print(f"Device {device}; training chips {labels.numel()}; held out {len(held_out)}", flush=True)

    weights = None if args.no_imagenet else models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, EUROSAT_NUM_CLASSES)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((labels.numel() + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(labels.numel())
        seen, correct, loss_sum = 0, 0, 0.0
        for start in range(0, labels.numel(), args.batch_size):
            idx = order[start:start + args.batch_size]
            x = chips[idx]
            # Flip / 90-degree rotation augmentation (nadir imagery has no canonical orientation)
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
            x = preprocess_chips(x.to(device))
            y = labels[idx].to(device)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.numel()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss_sum / seen:.4f}  train acc {correct / seen:.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    # Accuracy on the main campaign's held-out evaluation chips (reported, not tuned on)
    model.eval()
    splits = load_eurosat_splits(args.data_dir, calib_per_class=10, eval_per_class=15, seed=0)
    with torch.no_grad():
        preds = torch.cat([model(chunk.to(device)).argmax(1).cpu() for chunk in splits.eval_images.split(64)])
    eval_acc = (preds == splits.eval_labels).float().mean().item()
    print(f"held-out eval accuracy (10:15 split, {splits.eval_labels.numel()} chips): {eval_acc:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, args.out)
    meta = {
        "arch": "mobilenet_v2",
        "init": "random" if args.no_imagenet else "IMAGENET1K_V2",
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "optimizer": "AdamW+OneCycle",
        "augmentation": "hflip, rot90",
        "train_chips": int(labels.numel()),
        "exclude_splits": [list(s) for s in args.exclude_splits],
        "eval_accuracy_10_15": round(eval_acc, 4),
        "held_out_chips": sorted(held_out),
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    from aegis.runtime import configure_torch_threads

    configure_torch_threads()
    main()
