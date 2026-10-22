"""
Single-fault detection study: how well do the Aegis-v2 tiers see one persistent weight upset?

Each trial flips one stored bit, placed like the campaign injector (block and tensor proportional to bit
count, then a random element), in a chosen bit field. It then runs one monitored 16-image batch and records:
  - which behavioral tiers fired (T1 envelope, T2 BN drift) and the candidate blocks,
  - which blocks T3 confirmed and whether the struck block was among them (correct localization),
  - the accuracy loss on the labeled evaluation set, and the predictions (for class-confusion analysis).
The model is reset to its clean weights after every trial, so trials are independent.

Bit fields: exp_msb (bit 30), exponent (23-29), sign (31), mantissa_hi (16-22), mantissa_lo (0-15).

Example:
    python experiments/detection_study.py --arch resnet18 --trials-per-field 40
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.config import EUROSAT_CLASSES  # noqa: E402
from aegis.data.eurosat import load_eurosat_splits  # noqa: E402
from aegis.fault_engine.injector import ModelFaultInjector  # noqa: E402
from aegis.models.factory import load_backbone  # noqa: E402
from aegis.models.layer_registry import LayerRegistry  # noqa: E402
from aegis.monitor.orchestrator import AegisHealthMonitor  # noqa: E402

FIELDS = {
    "exp_msb": [30],
    "exponent": list(range(23, 30)),
    "sign": [31],
    "mantissa_hi": list(range(16, 23)),
    "mantissa_lo": list(range(0, 16)),
}


def predict(model, images, chunk=50):
    return torch.cat([model(c).argmax(1) for c in images.split(chunk)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    p.add_argument("--trials-per-field", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-per-class", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"results/detection_study_{args.arch}.json"
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model = load_backbone(args.arch)
    registry = LayerRegistry(model)
    splits = load_eurosat_splits(calib_per_class=10, eval_per_class=args.eval_per_class, seed=0)
    clean_state = {k: v.clone() for k, v in model.state_dict().items()}
    injector = ModelFaultInjector(model, registry, rng=np.random.default_rng(args.seed))

    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(splits.calib_images)

    labels = splits.eval_labels
    clean_pred = predict(model, splits.eval_images)
    clean_acc = (clean_pred == labels).float().mean().item()
    blocks = registry.block_ids
    bits = np.array([registry.get_block_metadata(b).total_bits_fp32 for b in blocks], dtype=np.float64)
    block_p = bits / bits.sum()
    n_eval = labels.numel()

    rows = []
    t0 = time.time()
    for field, positions in FIELDS.items():
        for k in range(args.trials_per_field):
            block = blocks[int(injector.rng.choice(len(blocks), p=block_p))]
            param = injector._sample_param_uniform_over_bits(registry.get_block_metadata(block))
            bit = rng.choice(positions)
            rec = injector.inject_targeted_bit_flip(block, param, None, bit)

            start = (len(rows) * args.batch) % (n_eval - args.batch)
            monitor.clear()
            monitor.attach()
            model(splits.eval_images[start:start + args.batch])
            monitor.detach()
            report = monitor.diagnose_current_state()
            t1 = sorted({monitor._map_layer_name_to_block_id(n) for n in report.tier1_violations})
            t2 = sorted({monitor._map_layer_name_to_block_id(n)
                         for n, d in report.tier2_drifts.items() if d.get("is_anomalous")})

            pred = predict(model, splits.eval_images)
            acc = (pred == labels).float().mean().item()
            to = rec.post_value
            rows.append({
                "field": field,
                "bit": bit,
                "block": block,
                "param": param,
                "from": rec.pre_value,
                "to": to if np.isfinite(to) else str(to),
                "t1": bool(t1),
                "t2": bool(t2),
                "t1_blocks": t1,
                "t2_blocks": t2,
                "confirmed": report.tier3_corrupted_blocks,
                "localized": block in report.tier3_corrupted_blocks,
                "acc": acc,
                "drop": clean_acc - acc,
                "pred_hist": torch.bincount(pred, minlength=len(EUROSAT_CLASSES)).tolist(),
                "confusion": torch.bincount(labels * len(EUROSAT_CLASSES) + pred,
                                            minlength=len(EUROSAT_CLASSES) ** 2).tolist(),
            })
            model.load_state_dict(clean_state)
            injector._fault_history.clear()
        print(f"{field}: {args.trials_per_field} trials, {time.time() - t0:.0f}s", flush=True)

    # Per-field summary. "Damaging" = loses more than one evaluation chip.
    chip = 1.0 / n_eval
    summary = {"arch": args.arch, "clean_accuracy": clean_acc, "eval_chips": n_eval, "fields": {}}
    for field in FIELDS:
        r = [x for x in rows if x["field"] == field]
        dmg = [x for x in r if x["drop"] > chip + 1e-9]
        alarm = [x for x in r if x["t1"] or x["t2"]]
        summary["fields"][field] = {
            "trials": len(r),
            "damaging": len(dmg),
            "mean_drop": float(np.mean([x["drop"] for x in r])),
            "max_drop": float(max(x["drop"] for x in r)),
            "alarm_rate": len(alarm) / len(r),
            "t1_rate": sum(x["t1"] for x in r) / len(r),
            "t2_rate": sum(x["t2"] for x in r) / len(r),
            "recall_damaging": (sum(x["t1"] or x["t2"] for x in dmg) / len(dmg)) if dmg else None,
            "localized_given_alarm": (sum(x["localized"] for x in alarm) / len(alarm)) if alarm else None,
            "false_restore_blocks": sum(len(set(x["confirmed"]) - {x["block"]}) for x in r),
            "silent_damaging": [round(x["drop"], 4) for x in dmg if not (x["t1"] or x["t2"])],
        }
    summary["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "classes": list(EUROSAT_CLASSES), "clean_pred_hist":
                   torch.bincount(clean_pred, minlength=len(EUROSAT_CLASSES)).tolist(), "trials": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
