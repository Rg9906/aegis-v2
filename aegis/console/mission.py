"""
Live Orbit Mission for the Console.

A lighter, streaming counterpart of MissionSimulationHarness for interactive use: four replicas
(unmitigated, periodic scrubbing, Aegis-v2, Aegis-v2 + CRC audit) receive the identical seeded
fault stream on the real Sentinel-2A track, and every step is emitted as it completes.

Same per-step protocol as the harness (inject -> scrub timer -> Aegis observes one unlabeled
incoming batch and restores CRC-confirmed blocks -> audit CRC scan on its period -> evaluate).
Differences, all disclosed in the UI: fewer strategies, a smaller labeled probe set, and an
optional radiation multiplier (>1x accelerates the Alsat-1-calibrated rates for demonstration).

Speed: replicas with bit-identical weights have identical accuracy (and identical monitor
reports), so results are cached by CRC fingerprint; before the first repair all four replicas
share one evaluation.
"""

import copy
import dataclasses
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from aegis.baselines.periodic_scrubbing import PeriodicScrubbingController
from aegis.config import DEFAULT_RAD_CONFIG
from aegis.console.common import Emit, block_crcs, hit_dict
from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.fault_engine.orbit_model import OrbitTrackRadiationEnvironment
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor, HealthReport
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.selective_restore import DetectionGatedRestorer

STRATEGIES: Tuple[Tuple[str, str], ...] = (
    ("unmitigated", "Unprotected"),
    ("scrub", "Periodic scrubbing"),
    ("aegis", "Aegis-v2"),
    ("audit", "Aegis-v2 + CRC audit"),
)
MAX_HITS_PER_EVENT = 80


@dataclass
class MissionConfig:
    hours: float = 12.0
    step_minutes: float = 60.0
    flux: float = 10.0
    scrub_hours: float = 6.0
    audit_hours: float = 6.0
    probe_per_class: int = 2
    seed: int = 0

    @property
    def total_steps(self) -> int:
        return int((self.hours * 60.0) // self.step_minutes)

    def validate(self) -> None:
        checks = [
            (1 <= self.hours <= 96, "hours must be within 1-96"),
            (15 <= self.step_minutes <= 360, "step_minutes must be within 15-360"),
            (1 <= self.total_steps <= 96, "a mission must have 1-96 steps"),
            (1 <= self.flux <= 100, "flux multiplier must be within 1-100"),
            (1 <= self.scrub_hours <= 48, "scrub_hours must be within 1-48"),
            (1 <= self.audit_hours <= 48, "audit_hours must be within 1-48"),
            (1 <= self.probe_per_class <= 15, "probe_per_class must be within 1-15"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)


class _Replica:
    """One strategy's copy of the payload model with its own repair machinery."""

    def __init__(self, key: str, base: nn.Module, calibrated: Optional[AegisHealthMonitor],
                 with_store: bool, injector_seed: Optional[int] = None):
        self.key = key
        self.model = copy.deepcopy(base).eval()
        self.registry = LayerRegistry(self.model)
        if injector_seed is None:
            self.injector = ModelFaultInjector(self.model, self.registry)
        else:
            self.injector = ModelFaultInjector(
                self.model, self.registry,
                generator=torch.Generator().manual_seed(injector_seed),
                rng=np.random.default_rng(injector_seed),
            )
        self.restorer = (DetectionGatedRestorer(self.registry, CompressedGoldenStore(self.registry))
                         if with_store else None)
        self.monitor: Optional[AegisHealthMonitor] = None
        if calibrated is not None:
            self.monitor = AegisHealthMonitor(self.model, self.registry)
            self.monitor.copy_calibration_from(calibrated)
            self.monitor.attach()
        self.acc_history: List[float] = []
        self.repairs = 0
        self.alarms = 0

    @property
    def bytes(self) -> int:
        return self.restorer.cumulative_bytes_transferred if self.restorer else 0

    def detach(self) -> None:
        if self.monitor is not None:
            self.monitor.detach()


class LiveMission:
    """Runs one mission on the calling thread; stop() makes it end after the current step."""

    def __init__(
        self,
        base_model: nn.Module,
        calibrated_monitor: AegisHealthMonitor,
        golden_crcs: Dict[str, int],
        probe_images: torch.Tensor,
        probe_labels: torch.Tensor,
        incoming_images: torch.Tensor,
        config: MissionConfig,
        emit: Emit,
        monitor_batch_size: int = 16,
    ):
        config.validate()
        self.config = config
        self.emit = emit
        self.golden_crcs = golden_crcs
        self.probe_images = probe_images
        self.probe_labels = probe_labels
        self.incoming = incoming_images
        self.monitor_batch_size = monitor_batch_size
        self._stop = threading.Event()

        self.replicas: Dict[str, _Replica] = {
            "unmitigated": _Replica("unmitigated", base_model, None, with_store=False, injector_seed=config.seed),
            "scrub": _Replica("scrub", base_model, None, with_store=True),
            "aegis": _Replica("aegis", base_model, calibrated_monitor, with_store=True),
            "audit": _Replica("audit", base_model, calibrated_monitor, with_store=True),
        }
        self.scrub_controller = PeriodicScrubbingController(
            self.replicas["scrub"].restorer, scrub_interval_hours=config.scrub_hours)
        rad = dataclasses.replace(
            DEFAULT_RAD_CONFIG,
            background_rate_per_bit_day=DEFAULT_RAD_CONFIG.background_rate_per_bit_day * config.flux,
            saa_peak_rate_per_bit_day=DEFAULT_RAD_CONFIG.saa_peak_rate_per_bit_day * config.flux,
        )
        self.env = OrbitTrackRadiationEnvironment(rad, random_seed=config.seed)
        # Separate stream so placing hits on the map never changes the fault stream itself
        self._place_rng = np.random.default_rng(config.seed + 1_000_003)
        registry = self.replicas["unmitigated"].registry
        self.total_bits = sum(registry.get_block_metadata(b).total_bits_fp32 for b in registry.block_ids)
        self.expected_flips_per_day = rad.measured_mean_rate_per_bit_day * self.total_bits * config.flux
        self.full_model_bytes = self.replicas["scrub"].restorer.golden_store.total_compressed_bytes
        self._acc_cache: Dict[Tuple[int, ...], float] = {}
        self.steps: List[Dict[str, Any]] = []
        self.summary: Optional[Dict[str, Any]] = None
        self.flips_total = 0
        self.msb_total = 0
        self.audits = 0

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # ---- measurement ------------------------------------------------------------------------

    def _fingerprint(self, rep: _Replica) -> Tuple[int, ...]:
        crcs = block_crcs(rep.registry)
        return tuple(crcs[b] for b in rep.registry.block_ids)

    def _accuracy(self, rep: _Replica, fingerprint: Tuple[int, ...]) -> float:
        if fingerprint not in self._acc_cache:
            correct = 0
            with torch.inference_mode():
                for start in range(0, self.probe_labels.numel(), 32):
                    logits = rep.model(self.probe_images[start:start + 32])
                    correct += int((logits.argmax(-1) == self.probe_labels[start:start + 32]).sum())
            if rep.monitor is not None:
                rep.monitor.clear()  # evaluation is measurement, not an observed mission batch
            self._acc_cache[fingerprint] = correct / max(1, self.probe_labels.numel())
        return self._acc_cache[fingerprint]

    def _observe(self, rep: _Replica, batch: torch.Tensor) -> HealthReport:
        rep.monitor.clear()
        with torch.inference_mode():
            rep.model(batch)
        report = rep.monitor.diagnose_current_state(run_tier3_scan=False)
        rep.monitor.clear()
        return report

    def _strike_minutes(self, start_min: float, end_min: float, n: int) -> List[float]:
        """When in the step each upset struck, for the map. The flux model draws a count per step,
        not per-upset times, so times are sampled from the step's per-minute rate profile (an
        SAA minute is ~75x likelier than a background one); illustrative, not recorded."""
        if n == 0:
            return []
        rates = self.env.minute_rates(start_min, end_min)
        first = int(np.floor(start_min))
        idx = self._place_rng.choice(rates.size, size=n, p=rates / rates.sum())
        t = first + idx + self._place_rng.random(n)
        return [round(float(x), 1) for x in np.clip(t, start_min, end_min)]

    def _incoming_batch(self, step_idx: int) -> torch.Tensor:
        n = self.incoming.size(0)
        idx = [(step_idx * self.monitor_batch_size + i) % n for i in range(self.monitor_batch_size)]
        return self.incoming[idx]

    # ---- protocol ---------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        steps = cfg.total_steps
        started = time.perf_counter()
        for step_idx in range(steps):
            if self._stop.is_set():
                break
            t0 = time.perf_counter()
            start_min = self.env.elapsed_minutes
            state = self.env.step(delta_minutes=cfg.step_minutes, total_model_bits=self.total_bits)
            end_min = self.env.elapsed_minutes

            # 1. Identical fault stream on every replica
            src = self.replicas["unmitigated"]
            records = src.injector.inject_orbital_flux_step(state, target_field="any")
            for rec in records:
                for key in ("scrub", "aegis", "audit"):
                    self.replicas[key].injector.inject_targeted_bit_flip(
                        rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)
            msb = sum(1 for r in records if r.bit_position == FLOAT32_EXPONENT_MSB)
            self.flips_total += len(records)
            self.msb_total += msb
            strike_min = self._strike_minutes(start_min, end_min, len(records))

            # 2. Blind scrubbing on its timer
            scrubbed = self.scrub_controller.step(end_min) is not None

            # 3. Aegis-v2 and Aegis+audit: observe the same unlabeled incoming batch, then repair.
            # Identical weights give identical reports, so the audit replica reuses Aegis's.
            batch = self._incoming_batch(step_idx)
            actions: Dict[str, Dict[str, Any]] = {k: {"alarm": False, "restored": []} for k in self.replicas}
            aegis, audit = self.replicas["aegis"], self.replicas["audit"]
            fp_aegis = self._fingerprint(aegis)
            report_aegis = self._observe(aegis, batch)
            report_audit = report_aegis if self._fingerprint(audit) == fp_aegis else self._observe(audit, batch)
            for rep, report in ((aegis, report_aegis), (audit, report_audit)):
                if not report.is_healthy:
                    rep.alarms += 1
                    event = rep.restorer.process_health_report(report, auto_recalibrate=False)
                    actions[rep.key]["alarm"] = True
                    actions[rep.key]["suspects"] = list(report.candidate_blocks)
                    if event is not None:
                        rep.repairs += 1
                        actions[rep.key]["restored"] = list(event.restored_blocks)
            period = cfg.audit_hours * 60.0
            audited = int(end_min // period) > int(start_min // period)
            if audited:
                self.audits += 1
                corrupt = audit.monitor.tier3_checksum.scan_all_blocks()
                if corrupt:
                    event = audit.restorer.process_health_report(
                        HealthReport(is_healthy=False, flagged_blocks=list(corrupt),
                                     primary_suspect_block=corrupt[0], tier3_corrupted_blocks=list(corrupt)),
                        auto_recalibrate=False)
                    audit.repairs += 1
                    actions["audit"]["restored"] = sorted(
                        set(actions["audit"]["restored"]) | set(event.restored_blocks),
                        key=audit.registry.block_ids.index)

            # 4. Evaluate (cached by fingerprint) and record which blocks are still corrupt
            per_strategy: Dict[str, Dict[str, Any]] = {}
            for key, rep in self.replicas.items():
                crcs = block_crcs(rep.registry)
                fp = tuple(crcs[b] for b in rep.registry.block_ids)
                acc = self._accuracy(rep, fp)
                rep.acc_history.append(acc)
                per_strategy[key] = {
                    "acc": round(acc, 4),
                    "mb": round(rep.bytes / 1e6, 3),
                    "corrupted": [b for b in rep.registry.block_ids if crcs[b] != self.golden_crcs[b]],
                    "alarm": actions[key]["alarm"],
                    "suspects": actions[key].get("suspects", []),
                    "restored": actions[key]["restored"],
                    "scrubbed": key == "scrub" and scrubbed,
                    "audited": key == "audit" and audited,
                }

            record = {
                "step": step_idx + 1,
                "total_steps": steps,
                "t_min": round(end_min, 1),
                "in_saa": bool(state.in_saa),
                "saa_minutes": round(float(state.saa_minutes_in_step), 1),
                "flux_x": round(float(state.instantaneous_flux_multiplier), 2),
                "flips": len(records),
                "msb": msb,
                "hits": [dict(hit_dict(r), t_min=m) for r, m in zip(records[:MAX_HITS_PER_EVENT], strike_min)],
                "hit_minutes": strike_min,
                "strategies": per_strategy,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
            self.steps.append(record)
            self.emit("mission.step", record)

        for rep in self.replicas.values():
            rep.detach()
        self.summary = self._summarize(stopped=self._stop.is_set(), wall_s=time.perf_counter() - started)
        return self.summary

    def _summarize(self, stopped: bool, wall_s: float) -> Dict[str, Any]:
        n = len(self.steps)
        hours = n * self.config.step_minutes / 60.0
        out = {"stopped": stopped, "steps": n, "hours": hours, "wall_s": round(wall_s, 1),
               "flips": self.flips_total, "msb": self.msb_total, "audits": self.audits,
               "scrubs": self.scrub_controller.total_scrubs_executed, "strategies": {}}
        for key, label in STRATEGIES:
            rep = self.replicas[key]
            hist = rep.acc_history
            out["strategies"][key] = {
                "label": label,
                "mean_acc": round(float(np.mean(hist)), 4) if hist else None,
                "final_acc": round(hist[-1], 4) if hist else None,
                "min_acc": round(float(np.min(hist)), 4) if hist else None,
                "mb": round(rep.bytes / 1e6, 3),
                "mb_per_day": round(rep.bytes / 1e6 / (hours / 24.0), 2) if hours > 0 else 0.0,
                # Scrub repairs are its timer-driven full reloads
                "repairs": self.scrub_controller.total_scrubs_executed if key == "scrub" else rep.repairs,
                "alarms": rep.alarms,
            }
        return out
