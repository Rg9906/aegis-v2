"""
Transient activation-fault experiment.

SEUs can also strike activation buffers during inference. Those faults vanish on the next
inference, so a correct repair system must NOT reload weights for them. Each trial flips one
bit of one activation element in a random conv/linear output for one incoming batch and records:
  - whether Tier 1/2 raised a behavioral alarm,
  - whether Tier 3 confirmed any block (would trigger a needless restore),
  - batch accuracy without and with Ranger clamping.

Example:
    python experiments/transient_faults.py --trials 60 --bit 30
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.baselines.ranger_clipper import RangerActivationClipper  # noqa: E402
from aegis.data.eurosat import load_eurosat_splits  # noqa: E402
from aegis.fault_engine.ieee754 import inject_single_bit_flip  # noqa: E402
from aegis.models.eurosat_classifier import EuroSATResNet18  # noqa: E402
from aegis.monitor.orchestrator import AegisHealthMonitor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--bit", type=int, default=30, help="Bit to flip (30 = exponent MSB, -1 = random)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/transient_faults.json")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    rng = random.Random(args.seed)

    classifier = EuroSATResNet18(pretrained=True)
    model, registry = classifier.backbone, classifier.registry
    splits = load_eurosat_splits(calib_per_class=10, eval_per_class=10, seed=0)
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    # The fault hook must run before monitor/Ranger hooks, so it is registered first
    target = {"name": None, "bit": None}

    def make_fault_hook(name):
        def hook(module, inp, out):
            if target["name"] != name:
                return None
            out = out.clone()
            index = rng.randrange(out.numel())
            inject_single_bit_flip(out, index, target["bit"])
            return out
        return hook

    for name, module in layers:
        module.register_forward_hook(make_fault_hook(name))

    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(splits.calib_images)
    monitor.attach()
    ranger = RangerActivationClipper()
    ranger.calibrate(model, splits.calib_images)

    n_eval = splits.eval_labels.numel()
    rows = []
    for trial in range(args.trials):
        start = (trial * args.batch) % (n_eval - args.batch)
        x = splits.eval_images[start:start + args.batch]
        y = splits.eval_labels[start:start + args.batch]
        target["name"] = rng.choice(layers)[0]
        target["bit"] = args.bit if args.bit >= 0 else rng.randrange(32)

        monitor.clear()
        acc_plain = (model(x).argmax(1) == y).float().mean().item()
        report = monitor.diagnose_current_state()
        ranger.attach(model)
        acc_ranger = (model(x).argmax(1) == y).float().mean().item()
        ranger.detach()

        struck_layer, struck_bit = target["name"], target["bit"]
        target["name"] = None
        clean = (model(x).argmax(1) == y).float().mean().item()
        rows.append({
            "layer": struck_layer,
            "bit": struck_bit,
            "behavioral_alarm": bool(report.tier1_violations) or any(
                d["is_anomalous"] for d in report.tier2_drifts.values()),
            "crc_confirmed_blocks": report.tier3_corrupted_blocks,
            "acc_clean": clean,
            "acc_faulty": acc_plain,
            "acc_faulty_ranger": acc_ranger,
        })

    n = len(rows)
    summary = {
        "trials": n,
        "bit": args.bit,
        "behavioral_alarm_rate": sum(r["behavioral_alarm"] for r in rows) / n,
        "spurious_restore_rate": sum(bool(r["crc_confirmed_blocks"]) for r in rows) / n,
        "mean_acc_clean": sum(r["acc_clean"] for r in rows) / n,
        "mean_acc_faulty": sum(r["acc_faulty"] for r in rows) / n,
        "mean_acc_faulty_ranger": sum(r["acc_faulty_ranger"] for r in rows) / n,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trials": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    from aegis.runtime import configure_torch_threads

    configure_torch_threads()
    main()
