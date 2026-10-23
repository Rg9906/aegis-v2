"""
Console Engine: an operator session over the real Aegis-v2 pipeline.

Holds one onboard payload model (with live T1/T2 hooks, a T3 CRC reference, a lossless golden
store and the detection-gated restorer) plus an untouched golden copy for side-by-side
predictions. Operators can strike chosen bits, run health checks, repair, scrub, classify real
EuroSAT chips, and launch live orbit missions. Every number is computed on request.

Thread safety: all commands take one lock. While a mission runs (on its own replicas) payload
commands are refused with EngineBusy so the mission keeps the CPU.
"""

import copy
import io
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from aegis.config import DEFAULT_EUROSAT_DATA_DIR, EUROSAT_CLASSES
from aegis.console.common import Emit, block_crcs, corrupted_blocks, finite, hit_dict
from aegis.console.mission import STRATEGIES, LiveMission, MissionConfig
from aegis.data.eurosat import _load_chips, preprocess_chips, split_eurosat_paths
from aegis.fault_engine.ieee754 import sample_bit_positions
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.factory import ARCHITECTURES, DEFAULT_MOBILENET_CHECKPOINT, assert_held_out, load_backbone
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor, HealthReport
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.selective_restore import DetectionGatedRestorer, RepairEvent

# Same split as the final campaigns (10 calibration + 15 evaluation chips per class, seed 0);
# for MobileNetV2 these chips are verified to be outside its training set.
CALIB_PER_CLASS = 10
EVAL_PER_CLASS = 15
FIELDS = ("exponent_msb", "exponent", "mantissa", "sign", "any")
MAX_STRIKE = 64


class EngineBusy(RuntimeError):
    """A mission (or another command) currently owns the payload."""


class EngineNotReady(RuntimeError):
    """The engine is still booting or failed to boot."""


def _empty_payload() -> Dict[str, Any]:
    return {"faults": [], "restored_at": None}


class ConsoleEngine:
    def __init__(self, emit: Optional[Emit] = None, data_dir: str = DEFAULT_EUROSAT_DATA_DIR,
                 probe_per_class: int = 3, seed: int = 0):
        self._emit_fn = emit or (lambda _t, _p: None)
        self.data_dir = data_dir
        self.probe_per_class = probe_per_class
        self.seed = seed
        self._lock = threading.RLock()
        self.status = "idle"
        self.boot_error: Optional[str] = None
        self.boot_stage = ""
        self.arch: Optional[str] = None
        self.field: Optional[Dict[str, Any]] = None
        self._mission: Optional[LiveMission] = None
        self._mission_thread: Optional[threading.Thread] = None
        self._mission_active = False
        self._mission_meta: Optional[Dict[str, Any]] = None
        self._mission_counter = 0
        self._incoming_cursor = 0

    # ---- events -----------------------------------------------------------------------------

    def emit(self, kind: str, payload: Dict[str, Any]) -> None:
        try:
            self._emit_fn(kind, payload)
        except Exception:  # a broken listener must never break the engine
            pass

    def log(self, text: str, level: str = "info", tag: str = "SYS") -> None:
        self.emit("log", {"t": time.time(), "level": level, "tag": tag, "text": text})

    # ---- boot -------------------------------------------------------------------------------

    def available_archs(self) -> List[str]:
        return [a for a in ARCHITECTURES if a != "mobilenet_v2" or os.path.exists(DEFAULT_MOBILENET_CHECKPOINT)]

    def boot(self, arch: str = "resnet18") -> None:
        """Loads model, data and monitors. Blocking (~15 s on a laptop CPU); emits progress."""
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch}'")
        with self._lock:
            if self.mission_running:
                raise EngineBusy("A mission is running")
            self.status = "booting"
            self.boot_error = None
            try:
                self._boot(arch)
                self.status = "ready"
                self.emit("boot", {"stage": "ready", "progress": 1.0, "arch": arch})
                self.log(f"Payload online: {self.arch_label} with {len(self.registry.block_ids)} restorable blocks, "
                         f"clean accuracy {self.clean_probe_acc:.1%} on {self.probe_labels.numel()} probe chips",
                         "ok")
                self.emit("state", self.snapshot())
            except Exception as exc:
                self.status = "error"
                self.boot_error = f"{type(exc).__name__}: {exc}"
                self.emit("boot", {"stage": "error", "progress": 1.0, "error": self.boot_error})
                self.log(f"Boot failed: {self.boot_error}", "error")
                raise

    def _stage(self, text: str, progress: float) -> None:
        self.boot_stage = text
        self.emit("boot", {"stage": text, "progress": progress})
        self.log(text, tag="BOOT")

    def _boot(self, arch: str) -> None:
        label = "MobileNetV2" if arch == "mobilenet_v2" else "ResNet-18"
        self._stage(f"Loading {label} EuroSAT weights", 0.05)
        model = load_backbone(arch).eval()

        self._stage("Loading Sentinel-2 EuroSAT chips (disjoint calibration / evaluation)", 0.15)
        calib_paths, eval_items = split_eurosat_paths(self.data_dir, CALIB_PER_CLASS, EVAL_PER_CLASS, self.seed)
        if arch == "mobilenet_v2":
            assert_held_out(DEFAULT_MOBILENET_CHECKPOINT, calib_paths + [p for p, _ in eval_items])
        calib_raw = _load_chips([os.path.join(self.data_dir, p) for p in calib_paths])
        eval_raw = _load_chips([os.path.join(self.data_dir, p) for p, _ in eval_items])
        calib_images = preprocess_chips(calib_raw)
        eval_images = preprocess_chips(eval_raw)
        eval_labels = torch.tensor([lab for _, lab in eval_items], dtype=torch.long)

        self._stage("Partitioning into restorable blocks; compressing golden store (zstd)", 0.3)
        golden_model = copy.deepcopy(model).eval()
        registry = LayerRegistry(model)
        store = CompressedGoldenStore(registry)

        self._stage(f"Calibrating T1 activation envelope + T2 BN drift on {calib_images.size(0)} clean chips", 0.4)
        monitor = AegisHealthMonitor(model, registry)
        monitor.calibrate(calib_images)
        monitor.attach()

        self._stage("Computing T3 CRC-32 references and measuring clean accuracy", 0.85)
        probe_idx: List[int] = []
        for cls in range(len(EUROSAT_CLASSES)):
            probe_idx += [i for i, (_, lab) in enumerate(eval_items) if lab == cls][: self.probe_per_class]
        probe_idx.sort()

        self.arch = arch
        self.arch_label = label
        self.model = model
        self.golden_model = golden_model
        self.registry = registry
        self.store = store
        self.monitor = monitor
        self.restorer = DetectionGatedRestorer(registry, store)
        self.injector = ModelFaultInjector(
            model, registry, generator=torch.Generator().manual_seed(self.seed),
            rng=np.random.default_rng(self.seed))
        self.calib_images = calib_images
        self.eval_raw = eval_raw
        self.eval_images = eval_images
        self.eval_labels = eval_labels
        self.eval_paths = [p for p, _ in eval_items]
        self.probe_idx = probe_idx
        self.probe_images = eval_images[probe_idx]
        self.probe_labels = eval_labels[probe_idx]
        # One chip per class to watch predictions change (first probe chip of each class)
        self.watch_idx = [next(i for i in probe_idx if int(eval_labels[i]) == c) for c in range(len(EUROSAT_CLASSES))]
        self.golden_crcs = block_crcs(registry)
        self._golden_probs: Dict[int, torch.Tensor] = {}
        self._reset_counters()
        # The live payload is still bit-identical to golden here, so this is the clean baseline
        self.clean_probe_acc = 0.0
        boot = self._measure("boot")
        self.clean_probe_acc = self.live_probe_acc = boot["accuracy"]
        self.boot_watch = self.last_watch = boot["watch"]

        if self.field is None:
            self._stage("Evaluating the IGRF geomagnetic field for the South Atlantic Anomaly", 0.95)
            from aegis.fault_engine.orbit_model import field_grid

            self.field = field_grid(alt_km=790.0)

    def _reset_counters(self) -> None:
        self.block_log: Dict[str, Dict[str, Any]] = {b: _empty_payload() for b in self.registry.block_ids}
        self.armed = True
        self.strikes = 0
        self.repairs: List[Dict[str, Any]] = []
        self.restorer = DetectionGatedRestorer(self.registry, self.store)
        self.store.reset_telemetry()

    # ---- guards -----------------------------------------------------------------------------

    @property
    def mission_running(self) -> bool:
        # An explicit flag, not thread liveness: the mission thread publishes its final state while
        # still alive, and that snapshot must already say the mission is over.
        return self._mission_active

    @contextmanager
    def _command(self):
        """Runs one operator command: fails fast if not ready, busy, or another command is running."""
        self._require_ready()
        if not self._lock.acquire(timeout=0.5):
            raise EngineBusy("Another command is still running")
        try:
            self._require_ready()
            yield
        finally:
            self._lock.release()

    def _require_ready(self) -> None:
        if self.status != "ready":
            raise EngineNotReady(self.boot_error or f"Engine is {self.status}")
        if self.mission_running:
            raise EngineBusy("A live mission is running; stop it or wait for it to finish")

    # ---- snapshot ---------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        base = {
            "status": self.status,
            "boot_stage": self.boot_stage,
            "boot_error": self.boot_error,
            "archs": self.available_archs(),
            "threads": torch.get_num_threads(),
            "mission_running": self.mission_running,
        }
        if self.status != "ready":
            return base
        crcs = block_crcs(self.registry)
        blocks = []
        for b in self.registry.block_ids:
            meta = self.registry.get_block_metadata(b)
            faults = self.block_log[b]["faults"]
            blocks.append({
                "id": b,
                "params": meta.param_count,
                "bytes": meta.size_bytes_fp32,
                "crc_ok": crcs[b] == self.golden_crcs[b],
                "faults": len(faults),
                "worst": max((f["sev"] for f in faults), key=["minor", "moderate", "critical"].index, default=None),
                "recent": faults[-4:],
                "restored_at": self.block_log[b]["restored_at"],
            })
        return {
            **base,
            "arch": self.arch,
            "arch_label": self.arch_label,
            "armed": self.armed,
            "blocks": blocks,
            "clean_acc": self.clean_probe_acc,
            "live_acc": self.live_probe_acc,
            "watch": self.last_watch,
            "probe_size": int(self.probe_labels.numel()),
            "gallery_size": int(self.eval_labels.numel()),
            "strikes": self.strikes,
            "repairs": len(self.repairs),
            "bytes_restored": self.restorer.cumulative_bytes_transferred,
            "model_bytes": self.store.total_uncompressed_bytes,
            "golden_bytes": self.store.total_compressed_bytes,
            "corrupted": [b for b in self.registry.block_ids if crcs[b] != self.golden_crcs[b]],
        }

    def _publish_state(self) -> Dict[str, Any]:
        snap = self.snapshot()
        self.emit("state", snap)
        return snap

    # ---- measurement ------------------------------------------------------------------------

    def _forward(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        out = []
        with torch.inference_mode():
            for start in range(0, images.size(0), 32):
                out.append(model(images[start:start + 32]))
        return torch.cat(out)

    def _probe_accuracy(self, model: torch.nn.Module, hooks: bool = True):
        logits = self._forward(model, self.probe_images)
        if hooks:
            self.monitor.clear()  # evaluation is measurement, not an observed incoming batch
        probs = torch.softmax(logits.float(), dim=-1)
        preds = probs.argmax(-1)
        acc = float((preds == self.probe_labels).float().mean())
        return acc, (preds, probs)

    def _golden_probs_for(self, indices: List[int]) -> torch.Tensor:
        missing = [i for i in indices if i not in self._golden_probs]
        if missing:
            probs = torch.softmax(self._forward(self.golden_model, self.eval_images[missing]).float(), dim=-1)
            for i, p in zip(missing, probs):
                self._golden_probs[i] = p
        return torch.stack([self._golden_probs[i] for i in indices])

    def _measure(self, stage: str) -> Dict[str, Any]:
        """Live-payload accuracy on the probe set plus the watch strip, compared with golden."""
        t0 = time.perf_counter()
        acc, (preds, probs) = self._probe_accuracy(self.model)
        self.live_probe_acc = acc
        golden = self._golden_probs_for(self.watch_idx)
        pos = {idx: k for k, idx in enumerate(self.probe_idx)}
        watch = []
        for i, gp in zip(self.watch_idx, golden):
            p = probs[pos[i]]
            watch.append({
                "index": i,
                "label": int(self.eval_labels[i]),
                "pred": int(p.argmax()),
                "conf": finite(float(p.max()), 4),
                "golden_pred": int(gp.argmax()),
            })
        self.last_watch = watch
        return {"stage": stage, "accuracy": acc, "clean": self.clean_probe_acc,
                "correct": int(round(acc * self.probe_labels.numel())), "total": int(self.probe_labels.numel()),
                "watch": watch, "ms": round((time.perf_counter() - t0) * 1000)}

    def _incoming(self, n: int = 16) -> torch.Tensor:
        """Next unlabeled batch the payload 'acquires' (rotating over the evaluation stream)."""
        total = self.eval_images.size(0)
        idx = [(self._incoming_cursor + k) % total for k in range(n)]
        self._incoming_cursor = (self._incoming_cursor + n) % total
        return self.eval_images[idx]

    # ---- diagnosis / repair -----------------------------------------------------------------

    def _observe(self, full_crc: bool = False) -> Dict[str, Any]:
        """Runs one monitored incoming batch through the payload and summarizes the health report."""
        t0 = time.perf_counter()
        self.monitor.clear()
        with torch.inference_mode():
            self.model(self._incoming())
        report = self.monitor.diagnose_current_state(run_tier3_scan=full_crc)
        self.monitor.clear()
        ms = round((time.perf_counter() - t0) * 1000)
        return self._report_dict(report, ms)

    def _report_dict(self, report: HealthReport, ms: int) -> Dict[str, Any]:
        to_block = self.monitor._map_layer_name_to_block_id
        t1 = [{
            "layer": name, "block": to_block(name),
            "nan": bool(v.get("has_nan")), "inf": bool(v.get("has_inf")),
            "max": finite(v.get("max_val", 0.0), 4), "min": finite(v.get("min_val", 0.0), 4),
            "lo": finite(v["expected_bound"][0], 4), "hi": finite(v["expected_bound"][1], 4),
        } for name, v in report.tier1_violations.items()]
        t2 = [{"layer": name, "block": to_block(name), "score": finite(v["drift_score"], 4),
               "threshold": finite(v["threshold"], 4)}
              for name, v in report.tier2_drifts.items() if v.get("is_anomalous")]
        silent = [] if report.tier3_corrupted_blocks else corrupted_blocks(self.registry, self.golden_crcs)
        out = {
            "healthy": report.is_healthy,
            "suspects": list(report.candidate_blocks),
            "confirmed": list(report.tier3_corrupted_blocks),
            "primary": report.primary_suspect_block,
            "t1": t1[:32], "t1_count": len(t1), "t1_layers": len(self.monitor.tier1_envelope.bounds),
            "t2": t2[:32], "t2_count": len(t2), "t2_layers": len(report.tier2_drifts),
            "false_alarm": bool(report.candidate_blocks) and not report.tier3_corrupted_blocks,
            "silent": silent if report.is_healthy else [],
            "ms": ms,
        }
        out["_report"] = report
        return out

    def _restore(self, report: HealthReport, mode: str) -> Optional[Dict[str, Any]]:
        if mode == "scrub":
            event: Optional[RepairEvent] = self.restorer.restore_full_model()
        else:
            event = self.restorer.process_health_report(report, auto_recalibrate=False)
        if event is None:
            return None
        for b in event.restored_blocks:
            self.block_log[b] = {"faults": [], "restored_at": time.time()}
        crcs = block_crcs(self.registry)
        rec = {
            "mode": mode,
            "blocks": list(event.restored_blocks),
            "bytes": event.total_bytes_transferred,
            "energy_uj": finite(event.total_energy_uj, 4),
            "ms": finite(event.downtime_ms, 4),
            "full_bytes": self.store.total_compressed_bytes,
            # Restored blocks must be bit-exact again (lossless golden store)
            "verified": all(crcs[b] == self.golden_crcs[b] for b in event.restored_blocks),
            "still_corrupted": [b for b in self.registry.block_ids if crcs[b] != self.golden_crcs[b]],
        }
        self.repairs.append(rec)
        return rec

    @staticmethod
    def _public(report: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in report.items() if not k.startswith("_")}

    # ---- commands ---------------------------------------------------------------------------

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        with self._command():
            self.armed = bool(armed)
            self.log(f"Aegis autopilot {'ARMED: alarms trigger automatic repair' if armed else 'DISARMED: damage persists until you repair'}",
                     "ok" if armed else "warn", "CTRL")
            return self._publish_state()

    def strike(self, block: str = "random", field: str = "exponent_msb", count: int = 1,
               param: Optional[str] = None) -> Dict[str, Any]:
        """Flips `count` bits, then (if armed) runs the monitor and repairs. Emits each stage."""
        if field not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}")
        if not 1 <= int(count) <= MAX_STRIKE:
            raise ValueError(f"count must be within 1-{MAX_STRIKE}")
        with self._command():
            if block != "random" and block not in self.registry.block_ids:
                raise ValueError(f"Unknown block '{block}'")
            if param is not None and (block == "random" or param not in self.registry.get_block_metadata(block).state_dict_keys):
                raise ValueError(f"Parameter '{param}' is not in block '{block}'")
            blocks = [self.registry.get_block_metadata(b) for b in self.registry.block_ids]
            weights = np.asarray([m.total_bits_fp32 for m in blocks], dtype=np.float64)
            hits = []
            for _ in range(int(count)):
                meta = (blocks[int(self.injector.rng.choice(len(blocks), p=weights / weights.sum()))]
                        if block == "random" else self.registry.get_block_metadata(block))
                name = param or self.injector._sample_param_uniform_over_bits(meta)
                bit = int(sample_bit_positions(1, field, generator=self.injector.generator)[0])
                rec = self.injector.inject_targeted_bit_flip(meta.block_id, name, None, bit)
                hit = hit_dict(rec, detail=True)
                hits.append(hit)
                self.block_log[meta.block_id]["faults"].append(hit)
                self.log(f"{meta.block_id} · {name}[{rec.flat_index}] bit {bit} ({hit['field'].replace('_', ' ')}): "
                         f"{hit['from']} -> {hit['to']}", "crit" if hit["sev"] == "critical" else "warn", "STRIKE")
            self.strikes += len(hits)
            self.emit("strike", {"hits": hits, "armed": self.armed})
            self._publish_state()

            impact = self._measure("after_strike")
            self.emit("impact", impact)
            self.log(f"Payload accuracy {impact['accuracy']:.1%} (clean {self.clean_probe_acc:.1%}) on "
                     f"{impact['total']} probe chips", "crit" if impact["accuracy"] < self.clean_probe_acc - 0.05 else "info",
                     "EVAL")
            result: Dict[str, Any] = {"hits": hits, "impact": impact, "diagnosis": None, "repair": None, "recovered": None}
            if self.armed:
                result.update(self._diagnose_and_repair(mode="aegis"))
            result["state"] = self._publish_state()
            return result

    def _diagnose_and_repair(self, mode: str) -> Dict[str, Any]:
        diagnosis = self._observe()
        self.emit("diagnosis", self._public(diagnosis))
        self._log_diagnosis(diagnosis)
        out: Dict[str, Any] = {"diagnosis": self._public(diagnosis), "repair": None, "recovered": None}
        if not diagnosis["healthy"]:
            repair = self._restore(diagnosis["_report"], mode)
            if repair:
                self.emit("repair", repair)
                self.log(f"Restored {', '.join(repair['blocks'])} from the golden store: {repair['bytes'] / 1e6:.3f} MB "
                         f"in {repair['ms']} ms (full reload would be {repair['full_bytes'] / 1e6:.1f} MB); "
                         f"CRC {'verified' if repair['verified'] else 'MISMATCH'}", "ok", "REPAIR")
                self._publish_state()
                recovered = self._measure("after_repair")
                self.emit("recovered", recovered)
                self.log(f"Payload accuracy back to {recovered['accuracy']:.1%}", "ok", "EVAL")
                out.update(repair=repair, recovered=recovered)
        return out

    def _log_diagnosis(self, d: Dict[str, Any]) -> None:
        if d["confirmed"]:
            self.log(f"ALARM: T1 {d['t1_count']} layer violations, T2 {d['t2_count']} BN drifts -> suspects "
                     f"{', '.join(d['suspects'])}; T3 CRC confirms {', '.join(d['confirmed'])}", "crit", "DETECT")
        elif d["false_alarm"]:
            self.log(f"T1/T2 flagged {', '.join(d['suspects'])} but T3 CRC found no corrupted bits: false alarm absorbed, "
                     "no restore", "warn", "DETECT")
        elif d["silent"]:
            self.log(f"No behavioral alarm, yet CRC differs in {', '.join(d['silent'])}: sub-threshold corruption "
                     "(only a CRC audit catches it)", "warn", "DETECT")
        else:
            self.log(f"Healthy: {d['t1_layers']} activation envelopes and {d['t2_layers']} BN layers within bounds",
                     "ok", "DETECT")

    def diagnose(self, full_crc: bool = False) -> Dict[str, Any]:
        with self._command():
            d = self._observe(full_crc=full_crc)
            self.emit("diagnosis", self._public(d))
            self._log_diagnosis(d)
            return {"diagnosis": self._public(d), "state": self._publish_state()}

    def repair(self, mode: str = "aegis") -> Dict[str, Any]:
        """aegis: monitor-gated selective restore. audit: CRC-scan every block, restore mismatches.
        scrub: blind full reload (the baseline)."""
        if mode not in ("aegis", "audit", "scrub"):
            raise ValueError("mode must be aegis, audit or scrub")
        with self._command():
            if mode == "aegis":
                out = self._diagnose_and_repair("aegis")
            else:
                corrupt = corrupted_blocks(self.registry, self.golden_crcs)
                if mode == "audit":
                    self.log(f"CRC audit of all {len(self.registry.block_ids)} blocks: "
                             f"{', '.join(corrupt) if corrupt else 'no mismatches'}", "warn" if corrupt else "ok", "AUDIT")
                report = HealthReport(is_healthy=not corrupt, flagged_blocks=corrupt,
                                      primary_suspect_block=corrupt[0] if corrupt else None,
                                      tier3_corrupted_blocks=corrupt)
                repair = self._restore(report, mode) if (corrupt or mode == "scrub") else None
                out = {"diagnosis": None, "repair": repair, "recovered": None}
                if repair:
                    self.emit("repair", repair)
                    self.log(f"{'Full scrub' if mode == 'scrub' else 'Audit restore'}: {len(repair['blocks'])} blocks, "
                             f"{repair['bytes'] / 1e6:.2f} MB in {repair['ms']} ms", "ok", "REPAIR")
                    out["recovered"] = self._measure("after_repair")
                    self.emit("recovered", out["recovered"])
            out["state"] = self._publish_state()
            return out

    def reset(self) -> Dict[str, Any]:
        """Restores every block from the golden store and clears counters."""
        with self._command():
            self.restorer.restore_full_model()
            self._reset_counters()
            self.live_probe_acc = self.clean_probe_acc
            self.last_watch = self.boot_watch
            self.monitor.clear()
            self.log("Payload reset to golden weights; counters cleared", "ok", "CTRL")
            return self._publish_state()

    def evaluate(self, scope: str = "probe") -> Dict[str, Any]:
        """Accuracy of the live payload vs golden, per class. scope: probe (fast) or full gallery."""
        if scope not in ("probe", "full"):
            raise ValueError("scope must be probe or full")
        with self._command():
            t0 = time.perf_counter()
            idx = self.probe_idx if scope == "probe" else list(range(self.eval_labels.numel()))
            live = torch.softmax(self._forward(self.model, self.eval_images[idx]).float(), dim=-1)
            self.monitor.clear()
            golden = self._golden_probs_for(idx)
            labels = self.eval_labels[idx]
            lp, gp = live.argmax(-1), golden.argmax(-1)
            per_class = []
            for c, name in enumerate(EUROSAT_CLASSES):
                m = labels == c
                n = int(m.sum())
                per_class.append({"class": name, "n": n,
                                  "live": float((lp[m] == c).float().mean()) if n else None,
                                  "golden": float((gp[m] == c).float().mean()) if n else None})
            confusion = np.zeros((10, 10), dtype=int)
            for t, p in zip(labels.tolist(), lp.tolist()):
                confusion[t, p] += 1
            acc = float((lp == labels).float().mean())
            if scope == "probe":
                self.live_probe_acc = acc
            out = {"scope": scope, "n": len(idx), "live": acc, "golden": float((gp == labels).float().mean()),
                   "per_class": per_class, "confusion": confusion.tolist(),
                   "ms": round((time.perf_counter() - t0) * 1000)}
            self.log(f"Evaluated {len(idx)} chips: payload {acc:.1%} vs golden {out['golden']:.1%} "
                     f"({out['ms'] / 1000:.1f} s)", "info", "EVAL")
            self.emit("evaluation", out)
            self._publish_state()
            return out

    def classify(self, indices: List[int]) -> Dict[str, Any]:
        if not indices or len(indices) > 32:
            raise ValueError("classify 1-32 chips at a time")
        with self._command():
            n = int(self.eval_labels.numel())
            if any(not 0 <= int(i) < n for i in indices):
                raise ValueError(f"chip index out of range 0-{n - 1}")
            indices = [int(i) for i in indices]
            t0 = time.perf_counter()
            live = torch.softmax(self._forward(self.model, self.eval_images[indices]).float(), dim=-1)
            # Not a monitored observation: T2 tolerances are calibrated for 16-chip batches, so a
            # 1-chip batch would raise spurious drift alarms. Health checks use _observe().
            self.monitor.clear()
            ms = round((time.perf_counter() - t0) * 1000)
            golden = self._golden_probs_for(indices)

            def top(p: torch.Tensor):
                vals, cls = torch.topk(torch.nan_to_num(p, nan=0.0), 3)
                return [{"cls": int(c), "p": finite(float(v), 4)} for v, c in zip(vals, cls)]

            results = []
            for i, lp, gp in zip(indices, live, golden):
                finite_ok = bool(torch.isfinite(lp).all())
                results.append({
                    "index": i, "label": int(self.eval_labels[i]), "path": self.eval_paths[i],
                    "live": {"pred": int(lp.argmax()) if finite_ok else None, "top": top(lp), "finite": finite_ok},
                    "golden": {"pred": int(gp.argmax()), "top": top(gp)},
                })
            out = {"results": results, "ms": ms, "corrupted": corrupted_blocks(self.registry, self.golden_crcs)}
            wrong = sum(1 for r in results if r["live"]["pred"] != r["label"])
            self.log(f"Classified {len(results)} chip(s) in {ms} ms: {len(results) - wrong} correct",
                     "warn" if wrong else "info", "INFER")
            return out

    # ---- gallery ----------------------------------------------------------------------------

    def gallery(self) -> Dict[str, Any]:
        if self.status != "ready":
            raise EngineNotReady(self.boot_error or f"Engine is {self.status}")
        return {"classes": list(EUROSAT_CLASSES),
                "chips": [{"index": i, "label": int(lab), "probe": i in set(self.probe_idx)}
                          for i, lab in enumerate(self.eval_labels.tolist())],
                "watch": self.watch_idx}

    def chip_png(self, index: int, scale: int = 1) -> bytes:
        if self.status != "ready":
            raise EngineNotReady("Engine is not ready")
        if not 0 <= index < self.eval_raw.size(0):
            raise ValueError("chip index out of range")
        img = Image.fromarray(self.eval_raw[index].permute(1, 2, 0).numpy())
        if scale > 1:
            img = img.resize((64 * scale, 64 * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ---- missions ---------------------------------------------------------------------------

    def start_mission(self, config: MissionConfig) -> Dict[str, Any]:
        config.validate()
        with self._command():
            self._mission_counter += 1
            mission_id = self._mission_counter
            from aegis.fault_engine.orbit_model import track_payload

            probe_idx: List[int] = []
            for cls in range(len(EUROSAT_CLASSES)):
                probe_idx += [i for i, lab in enumerate(self.eval_labels.tolist()) if lab == cls][: config.probe_per_class]
            probe_idx.sort()
            mission = LiveMission(
                base_model=self.golden_model,
                calibrated_monitor=self.monitor,
                golden_crcs=self.golden_crcs,
                probe_images=self.eval_images[probe_idx],
                probe_labels=self.eval_labels[probe_idx],
                incoming_images=self.eval_images,
                config=config,
                emit=self.emit,
            )
            track = track_payload(config.total_steps * config.step_minutes + 100.0, step_minutes=2.0)
            meta = {
                "id": mission_id,
                "config": {**config.__dict__, "total_steps": config.total_steps},
                "strategies": [{"key": k, "label": label} for k, label in STRATEGIES],
                "blocks": [{"id": b, "bytes": self.registry.get_block_metadata(b).size_bytes_fp32}
                           for b in self.registry.block_ids],
                "arch": self.arch, "arch_label": self.arch_label,
                "clean_acc": mission._accuracy(mission.replicas["unmitigated"],
                                               mission._fingerprint(mission.replicas["unmitigated"])),
                "probe_size": len(probe_idx),
                "expected_flips_per_day": round(mission.expected_flips_per_day, 1),
                "full_model_mb": round(mission.full_model_bytes / 1e6, 2),
                "track": track,
                "field": self.field,
                "started": time.time(),
            }
            self._mission = mission
            self._mission_meta = meta
            self.emit("mission.start", meta)
            self.log(f"Mission #{mission_id} launched: {config.hours:g} h on the Sentinel-2A track, {config.total_steps} steps, "
                     f"radiation {config.flux:g}x ({meta['expected_flips_per_day']:.0f} expected flips/day), "
                     f"{len(probe_idx)} probe chips", "ok", "MISSION")
            if config.flux > 1:
                self.log(f"Radiation is accelerated {config.flux:g}x over the Alsat-1-calibrated rate for demonstration",
                         "warn", "MISSION")

            def run():
                try:
                    summary = mission.run()
                    meta["summary"] = summary
                    self.emit("mission.end", {"id": mission_id, **summary})
                    best = summary["strategies"]
                    self.log(f"Mission #{mission_id} {'stopped' if summary['stopped'] else 'complete'}: {summary['flips']} flips; "
                             + "; ".join(f"{best[k]['label']} {best[k]['mean_acc']:.1%} mean, {best[k]['mb']:.1f} MB"
                                         for k, _ in STRATEGIES if best[k]["mean_acc"] is not None),
                             "ok", "MISSION")
                except Exception as exc:  # surfaced to the UI instead of dying silently
                    meta["error"] = f"{type(exc).__name__}: {exc}"
                    self.emit("mission.error", {"id": mission_id, "error": meta["error"]})
                    self.log(f"Mission #{mission_id} failed: {meta['error']}", "error", "MISSION")
                finally:
                    self._mission_active = False
                    self.emit("state", self.snapshot())

            self._mission_active = True
            self._mission_thread = threading.Thread(target=run, name=f"mission-{mission_id}", daemon=True)
            self._mission_thread.start()
            return {k: v for k, v in meta.items() if k not in ("track", "field")}

    def stop_mission(self) -> Dict[str, Any]:
        if self._mission is None or not self.mission_running:
            return {"stopping": False}
        self._mission.stop()
        self.log("Stop requested: the mission ends after the current step", "warn", "MISSION")
        return {"stopping": True}

    def mission_state(self) -> Optional[Dict[str, Any]]:
        """Everything a reconnecting console needs to redraw the latest mission."""
        if self._mission_meta is None:
            return None
        return {**self._mission_meta, "steps": list(self._mission.steps), "running": self.mission_running,
                "stopping": self._mission.stopping}

    def wait_for_mission(self, timeout: Optional[float] = None) -> None:
        if self._mission_thread is not None:
            self._mission_thread.join(timeout)
