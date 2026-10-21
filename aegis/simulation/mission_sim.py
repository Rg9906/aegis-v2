"""
LEO Orbital Mission Simulation and Benchmark Campaign Substructure.

Executes time-stepped multi-orbit simulations modeling accumulated radiation flux,
SAA passes, and comparative resilience evaluation across all mitigation strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import copy
import numpy as np
import torch
import torch.nn as nn

from aegis.baselines.blink_recentering import BlinkRecenteringMitigator
from aegis.baselines.ground_uplink import GroundUplinkReloader
from aegis.baselines.periodic_scrubbing import PeriodicScrubbingController
from aegis.baselines.ranger_clipper import RangerActivationClipper
from aegis.config import DEFAULT_RAD_CONFIG, RadiationEnvironmentConfig
from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB, TargetFieldType
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.fault_engine.radiation_model import LEORadiationEnvironment
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.recalibrate import BNRecalibrator
from aegis.repair.selective_restore import DetectionGatedRestorer


@dataclass
class TimeStepTelemetry:
    """Telemetry recorded at each orbital simulation step."""
    step_index: int
    elapsed_minutes: float
    elapsed_days: float
    orbit_index: int
    in_saa: bool
    saa_minutes_in_step: float
    sbus_injected_step: int
    mbus_injected_step: int
    bit_flips_step: int
    exponent_msb_flips_step: int
    cumulative_faults: int
    acc_unmitigated: float
    acc_ranger: float
    acc_blink: float
    acc_periodic_scrub: float
    acc_aegis: float
    acc_ranger_aegis: float
    acc_uplink: float
    energy_aegis_uj: float
    energy_scrub_uj: float
    energy_ranger_aegis_uj: float
    energy_uplink_uj: float
    monitor_energy_aegis_uj: float
    monitor_compute_energy_aegis_uj: float
    bytes_aegis: int
    bytes_scrub: int
    bytes_ranger_aegis: int
    bytes_uplink: int
    scrub_executed: bool
    aegis_alarm: bool
    aegis_restored_blocks: List[str] = field(default_factory=list)
    # Extra periodic-scrubbing replicas for the accuracy-vs-bandwidth Pareto sweep (key: hours)
    acc_scrub_sweep: Dict[str, float] = field(default_factory=dict)
    bytes_scrub_sweep: Dict[str, int] = field(default_factory=dict)


class MissionSimulationHarness:
    """
    Coordinates side-by-side execution of all mitigation approaches
    under identical, physically grounded LEO radiation flux.

    Protocol per step:
      1. Advance orbit clock; sample Poisson upsets integrated over SAA/background minutes.
      2. Apply the identical bit flips to all five model replicas.
      3. Periodic scrubbing reloads the full model if its timer expired.
      4. Aegis-v2 observes one unlabeled incoming batch (label-free monitor) and, if alarmed,
         restores the CRC-confirmed blocks from the golden store. Ranger+Aegis does the same
         while also clamping activations; Ground uplink detects the same way but can only
         reload during ground-station contacts.
      5. All replicas are scored on the labeled evaluation set.

    Aegis-family energy = golden-store reads + DRAM writes + DRAM reads for T3 CRC scans
    + T1/T2 hook compute on each monitored batch (+ recalibration inference if enabled).
    Ranger/BLINK clamp and re-centering compute is not charged (they move no bytes).
    """

    def __init__(
        self,
        base_model: nn.Module,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
        calib_images: Optional[torch.Tensor] = None,
        rad_config: RadiationEnvironmentConfig = DEFAULT_RAD_CONFIG,
        random_seed: int = 42,
        fault_field: TargetFieldType = "any",
        scrub_interval_hours: float = 12.0,
        monitor_batch_size: int = 16,
        eval_chunk_size: int = 50,
        aegis_recalibrate: bool = False,
        scrub_sweep_hours: Sequence[float] = (),
        golden_int8: bool = False,
        orbit: str = "periodic",
        tle_path: Optional[str] = None,
    ):
        self.val_images = val_images
        self.val_labels = val_labels
        # Calibration imagery should be disjoint from evaluation imagery; fall back to a val
        # slice only for quick unit tests.
        calib_images = calib_images if calib_images is not None else val_images[:8]
        self.rad_config = rad_config
        self.fault_field = fault_field
        self.monitor_batch_size = monitor_batch_size
        self.eval_chunk_size = eval_chunk_size
        self.aegis_recalibrate = aegis_recalibrate
        # INT8 golden copy (Aegis + Ranger+Aegis only): ~4x smaller, lossy restores
        self.golden_int8 = golden_int8

        # 1. Models for each strategy (cloned independently)
        self.model_unmitigated = copy.deepcopy(base_model).eval()
        self.model_ranger = copy.deepcopy(base_model).eval()
        self.model_blink = copy.deepcopy(base_model).eval()
        self.model_scrub = copy.deepcopy(base_model).eval()
        self.model_aegis = copy.deepcopy(base_model).eval()

        # 2. Registries
        self.reg_unmitigated = LayerRegistry(self.model_unmitigated)
        self.reg_ranger = LayerRegistry(self.model_ranger)
        self.reg_blink = LayerRegistry(self.model_blink)
        self.reg_scrub = LayerRegistry(self.model_scrub)
        self.reg_aegis = LayerRegistry(self.model_aegis)

        # 3. Fault Injectors. Only the unmitigated injector samples faults (seeded); the
        # others replay its exact (param, index, bit) records.
        torch_gen = torch.Generator().manual_seed(random_seed)
        self.inj_unmitigated = ModelFaultInjector(
            self.model_unmitigated, self.reg_unmitigated,
            generator=torch_gen, rng=np.random.default_rng(random_seed),
        )
        self.inj_ranger = ModelFaultInjector(self.model_ranger, self.reg_ranger)
        self.inj_blink = ModelFaultInjector(self.model_blink, self.reg_blink)
        self.inj_scrub = ModelFaultInjector(self.model_scrub, self.reg_scrub)
        self.inj_aegis = ModelFaultInjector(self.model_aegis, self.reg_aegis)

        # 4. Mitigation Engines
        # "tle": real SGP4 ground track + IGRF low-field SAA; "periodic": fixed synthetic passes
        if orbit == "tle":
            from aegis.fault_engine.orbit_model import DEFAULT_TLE_PATH, OrbitTrackRadiationEnvironment

            self.rad_env = OrbitTrackRadiationEnvironment(
                rad_config, random_seed=random_seed, tle_path=tle_path or DEFAULT_TLE_PATH
            )
        elif orbit == "periodic":
            self.rad_env = LEORadiationEnvironment(rad_config, random_seed=random_seed)
        else:
            raise ValueError(f"Unknown orbit model '{orbit}'")
        self.orbit = orbit

        # Ranger Setup
        self.ranger_clipper = RangerActivationClipper()
        self.ranger_clipper.calibrate(self.model_ranger, calib_images)
        self.ranger_clipper.attach(self.model_ranger)

        # BLINK Setup
        self.blink_mitigator = BlinkRecenteringMitigator(drift_threshold=3.0)
        self.blink_mitigator.attach(self.model_blink)

        # Periodic Scrub Setup
        self.store_scrub = CompressedGoldenStore(self.reg_scrub)
        self.restorer_scrub = DetectionGatedRestorer(self.reg_scrub, self.store_scrub)
        self.scrub_controller = PeriodicScrubbingController(
            self.restorer_scrub, scrub_interval_hours=scrub_interval_hours
        )

        # Aegis-v2 Setup
        self.store_aegis = CompressedGoldenStore(self.reg_aegis, quantize_int8=golden_int8)
        self.recal_aegis = BNRecalibrator(buffer_capacity=64)
        self.recal_aegis.ingest_unlabeled_imagery(calib_images)
        self.restorer_aegis = DetectionGatedRestorer(self.reg_aegis, self.store_aegis, self.recal_aegis)
        self.monitor_aegis = AegisHealthMonitor(self.model_aegis, self.reg_aegis)
        self.monitor_aegis.calibrate(calib_images)
        self.monitor_aegis.attach()

        # Ranger + Aegis-v2: monitor hooks are registered BEFORE the clamping hooks, so the
        # monitor observes raw activations (forward hooks run in registration order) while
        # downstream layers receive clamped ones.
        self.model_combo = copy.deepcopy(base_model).eval()
        self.reg_combo = LayerRegistry(self.model_combo)
        self.inj_combo = ModelFaultInjector(self.model_combo, self.reg_combo)
        self.ranger_combo = RangerActivationClipper()
        self.ranger_combo.calibrate(self.model_combo, calib_images)
        self.store_combo = CompressedGoldenStore(self.reg_combo, quantize_int8=golden_int8)
        self.restorer_combo = DetectionGatedRestorer(self.reg_combo, self.store_combo)
        self.monitor_combo = AegisHealthMonitor(self.model_combo, self.reg_combo)
        self.monitor_combo.calibrate(calib_images)
        self.monitor_combo.attach()
        self.ranger_combo.attach(self.model_combo)

        # Ground uplink: same onboard detection, repair only at ground contacts
        self.model_uplink = copy.deepcopy(base_model).eval()
        self.reg_uplink = LayerRegistry(self.model_uplink)
        self.inj_uplink = ModelFaultInjector(self.model_uplink, self.reg_uplink)
        self.monitor_uplink = AegisHealthMonitor(self.model_uplink, self.reg_uplink)
        self.monitor_uplink.calibrate(calib_images)
        self.monitor_uplink.attach()
        self.uplink = GroundUplinkReloader(self.reg_uplink, CompressedGoldenStore(self.reg_uplink))

        # Optional additional scrubbing replicas at other intervals (identical fault stream)
        self.scrub_sweep = []
        for hours in scrub_sweep_hours:
            model = copy.deepcopy(base_model).eval()
            registry = LayerRegistry(model)
            restorer = DetectionGatedRestorer(registry, CompressedGoldenStore(registry))
            self.scrub_sweep.append({
                "key": f"{hours:g}",
                "model": model,
                "injector": ModelFaultInjector(model, registry),
                "restorer": restorer,
                "controller": PeriodicScrubbingController(restorer, scrub_interval_hours=hours),
            })

        # T1/T2 hook ops on monitored incoming batches only (evaluation passes through the same
        # hooks are measurement, not mission work, so they are excluded). Keyed by id(monitor).
        self._monitor_ops: Dict[int, int] = {}

        self.telemetry_history: List[TimeStepTelemetry] = []
        self.clean_accuracy: float = self._evaluate_model(self.model_unmitigated)

    def _evaluate_model(self, model: nn.Module) -> float:
        """Computes top-1 classification accuracy on validation set (chunked to bound memory)."""
        model.eval()
        device = next(model.parameters()).device
        correct = 0
        with torch.no_grad():
            for start in range(0, self.val_labels.numel(), self.eval_chunk_size):
                images = self.val_images[start:start + self.eval_chunk_size].to(device)
                labels = self.val_labels[start:start + self.eval_chunk_size].to(device)
                correct += (torch.argmax(model(images), dim=-1) == labels).sum().item()
        total = self.val_labels.numel()
        return float(correct / total if total > 0 else 0.0)

    def _incoming_batch(self, step_idx: int) -> torch.Tensor:
        """Unlabeled imagery the payload acquires this step (rotating window over the stream)."""
        n = self.val_images.size(0)
        idx = [(step_idx * self.monitor_batch_size + i) % n for i in range(self.monitor_batch_size)]
        return self.val_images[idx]

    def run_simulation(
        self,
        total_days: float = 3.0,
        step_minutes: float = 60.0,
        progress: bool = False,
    ) -> List[TimeStepTelemetry]:
        """
        Executes multi-day orbital campaign, logging telemetry across all strategies.
        """
        total_steps = int((total_days * 1440.0) // step_minutes)
        total_model_bits = sum(b.total_bits_fp32 for b in self.reg_aegis._blocks.values())

        for step_idx in range(total_steps):
            # 1. Advance orbital clock
            orbital_state = self.rad_env.step(delta_minutes=step_minutes, total_model_bits=total_model_bits)

            # 2. Inject identical faults across all 5 models
            records = self.inj_unmitigated.inject_orbital_flux_step(orbital_state, target_field=self.fault_field)
            for rec in records:
                # Mirror same bit-flip on all other models for fair comparison
                mirrors = [self.inj_ranger, self.inj_blink, self.inj_scrub, self.inj_aegis,
                           self.inj_combo, self.inj_uplink]
                mirrors += [rep["injector"] for rep in self.scrub_sweep]
                for inj in mirrors:
                    inj.inject_targeted_bit_flip(rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)

            # 3. Execute Periodic Scrubbing baseline step
            scrub_event = self.scrub_controller.step(self.rad_env.elapsed_minutes)
            for rep in self.scrub_sweep:
                rep["controller"].step(self.rad_env.elapsed_minutes)

            # 4. Aegis-v2: label-free monitoring of the incoming batch, then gated repair.
            # T3 checksums are only consulted for blocks that T1/T2 flag behaviorally.
            incoming = self._incoming_batch(step_idx)
            report = self._observe(self.monitor_aegis, self.model_aegis, incoming)

            aegis_event = None
            if not report.is_healthy:
                aegis_event = self.restorer_aegis.process_health_report(
                    report, auto_recalibrate=self.aegis_recalibrate
                )
                if self.golden_int8:
                    # Lossy restore: the restored block is the new known-good reference
                    self.monitor_aegis.rebaseline_blocks(aegis_event.restored_blocks)
                if self.aegis_recalibrate:
                    # Recalibration rewrote BN statistics everywhere; re-baseline T3 so the
                    # next alarm is not confirmed spuriously. (Not needed for lossless restore.)
                    self.monitor_aegis.refresh_golden_checksums()

            combo_report = self._observe(self.monitor_combo, self.model_combo, incoming)
            if not combo_report.is_healthy:
                combo_event = self.restorer_combo.process_health_report(combo_report, auto_recalibrate=False)
                if self.golden_int8:
                    self.monitor_combo.rebaseline_blocks(combo_event.restored_blocks)

            self.uplink.report(self._observe(self.monitor_uplink, self.model_uplink, incoming))
            self.uplink.step(self.rad_env.elapsed_minutes - step_minutes, self.rad_env.elapsed_minutes)

            # 5. Evaluate accuracies of all models
            accs = {
                name: self._evaluate_model(model)
                for name, model in (
                    ("unmitigated", self.model_unmitigated),
                    ("ranger", self.model_ranger),
                    ("blink", self.model_blink),
                    ("scrub", self.model_scrub),
                    ("aegis", self.model_aegis),
                    ("combo", self.model_combo),
                    ("uplink", self.model_uplink),
                )
            }

            # 6. Record telemetry
            step_record = TimeStepTelemetry(
                step_index=step_idx + 1,
                elapsed_minutes=round(self.rad_env.elapsed_minutes, 1),
                elapsed_days=round(self.rad_env.elapsed_days, 4),
                orbit_index=self.rad_env.current_orbit_index,
                in_saa=orbital_state.in_saa,
                saa_minutes_in_step=round(orbital_state.saa_minutes_in_step, 2),
                sbus_injected_step=orbital_state.total_sbus_generated,
                mbus_injected_step=orbital_state.total_mbus_generated,
                bit_flips_step=len(records),
                exponent_msb_flips_step=sum(1 for r in records if r.bit_position == FLOAT32_EXPONENT_MSB),
                cumulative_faults=self.inj_unmitigated.total_faults_injected,
                acc_unmitigated=round(accs["unmitigated"], 4),
                acc_ranger=round(accs["ranger"], 4),
                acc_blink=round(accs["blink"], 4),
                acc_periodic_scrub=round(accs["scrub"], 4),
                acc_aegis=round(accs["aegis"], 4),
                acc_ranger_aegis=round(accs["combo"], 4),
                acc_uplink=round(accs["uplink"], 4),
                energy_aegis_uj=round(
                    self.restorer_aegis.cumulative_energy_uj + self._monitor_energy_uj(self.monitor_aegis), 2),
                energy_scrub_uj=round(self.restorer_scrub.cumulative_energy_uj, 2),
                energy_ranger_aegis_uj=round(
                    self.restorer_combo.cumulative_energy_uj + self._monitor_energy_uj(self.monitor_combo), 2),
                energy_uplink_uj=round(
                    self.uplink.cumulative_energy_uj + self._monitor_energy_uj(self.monitor_uplink), 2),
                monitor_energy_aegis_uj=round(self._monitor_energy_uj(self.monitor_aegis), 2),
                monitor_compute_energy_aegis_uj=round(self._hook_compute_energy_uj(self.monitor_aegis), 2),
                bytes_aegis=self.restorer_aegis.cumulative_bytes_transferred,
                bytes_scrub=self.restorer_scrub.cumulative_bytes_transferred,
                bytes_ranger_aegis=self.restorer_combo.cumulative_bytes_transferred,
                bytes_uplink=self.uplink.cumulative_bytes,
                scrub_executed=scrub_event is not None,
                aegis_alarm=not report.is_healthy,
                aegis_restored_blocks=list(aegis_event.restored_blocks) if aegis_event else [],
                acc_scrub_sweep={
                    rep["key"]: round(self._evaluate_model(rep["model"]), 4) for rep in self.scrub_sweep
                },
                bytes_scrub_sweep={
                    rep["key"]: rep["restorer"].cumulative_bytes_transferred for rep in self.scrub_sweep
                },
            )
            self.telemetry_history.append(step_record)
            if progress:
                print(
                    f"  step {step_idx + 1:>4}/{total_steps}  day {step_record.elapsed_days:6.2f}  "
                    f"flips {step_record.bit_flips_step:>4} (msb {step_record.exponent_msb_flips_step})  "
                    f"acc U/R/B/S/A/RA/G = {accs['unmitigated']:.2f}/{accs['ranger']:.2f}/"
                    f"{accs['blink']:.2f}/{accs['scrub']:.2f}/{accs['aegis']:.2f}/"
                    f"{accs['combo']:.2f}/{accs['uplink']:.2f}  "
                    f"restored={step_record.aegis_restored_blocks}",
                    flush=True,
                )

        return self.telemetry_history

    def _observe(self, monitor: AegisHealthMonitor, model: nn.Module, batch: torch.Tensor):
        """Runs the incoming batch through a monitored replica and returns its health report."""
        monitor.clear()
        ops_before = monitor.tier1_envelope.ops_performed + monitor.tier2_bn_drift.ops_performed
        with torch.no_grad():
            model(batch)
        ops_after = monitor.tier1_envelope.ops_performed + monitor.tier2_bn_drift.ops_performed
        self._monitor_ops[id(monitor)] = self._monitor_ops.get(id(monitor), 0) + ops_after - ops_before
        return monitor.diagnose_current_state(run_tier3_scan=False)

    def _monitor_energy_uj(self, monitor: AegisHealthMonitor) -> float:
        """All monitoring energy so far: T3 CRC DRAM reads + T1/T2 hook compute."""
        pj_per_bit = self.store_aegis.budget_config.dram_read_energy_pj_per_bit
        crc_uj = monitor.tier3_checksum.bytes_scanned * 8 * pj_per_bit / 1e6
        return crc_uj + self._hook_compute_energy_uj(monitor)

    def _hook_compute_energy_uj(self, monitor: AegisHealthMonitor) -> float:
        """T1/T2 elementwise hook work on monitored batches so far."""
        ops = self._monitor_ops.get(id(monitor), 0)
        return ops * self.store_aegis.budget_config.monitor_energy_pj_per_op / 1e6

    def get_campaign_summary(self) -> Dict[str, object]:
        """Calculates mean (time-averaged) accuracy, total energy, and reliability metrics."""
        if not self.telemetry_history:
            return {}

        history = self.telemetry_history
        n_steps = len(history)

        def mean_of(attr: str) -> float:
            return round(sum(getattr(t, attr) for t in history) / n_steps, 4)

        last = history[-1]
        energy_savings_pct = (
            (last.energy_scrub_uj - last.energy_aegis_uj) / last.energy_scrub_uj * 100.0
            if last.energy_scrub_uj > 0
            else 0.0
        )

        return {
            "total_simulated_days": last.elapsed_days,
            "total_faults_accumulated": last.cumulative_faults,
            "exponent_msb_flips_total": sum(t.exponent_msb_flips_step for t in history),
            "fault_field": self.fault_field,
            "clean_accuracy": round(self.clean_accuracy, 4),
            "mean_accuracy": {
                "unmitigated": mean_of("acc_unmitigated"),
                "ranger_clipping": mean_of("acc_ranger"),
                "blink_recentering": mean_of("acc_blink"),
                "periodic_scrubbing": mean_of("acc_periodic_scrub"),
                "aegis_v2": mean_of("acc_aegis"),
                "ranger_aegis": mean_of("acc_ranger_aegis"),
                "ground_uplink": mean_of("acc_uplink"),
            },
            "final_accuracy": {
                "unmitigated": last.acc_unmitigated,
                "ranger_clipping": last.acc_ranger,
                "blink_recentering": last.acc_blink,
                "periodic_scrubbing": last.acc_periodic_scrub,
                "aegis_v2": last.acc_aegis,
                "ranger_aegis": last.acc_ranger_aegis,
                "ground_uplink": last.acc_uplink,
            },
            "energy_consumed_uj": {
                "periodic_scrubbing": last.energy_scrub_uj,
                "aegis_v2": last.energy_aegis_uj,
                "ranger_aegis": last.energy_ranger_aegis_uj,
                "ground_uplink": last.energy_uplink_uj,
                "aegis_v2_monitor_share": last.monitor_energy_aegis_uj,
                "aegis_v2_monitor_compute_share": last.monitor_compute_energy_aegis_uj,
                "energy_savings_percent": round(energy_savings_pct, 2),
            },
            "bytes_transferred": {
                "periodic_scrubbing": last.bytes_scrub,
                "aegis_v2": last.bytes_aegis,
                "ranger_aegis": last.bytes_ranger_aegis,
                "ground_uplink": last.bytes_uplink,
            },
            "ranger_aegis_restorations_count": self.restorer_combo.total_restorations_performed,
            "ground_uplink_contacts_used": len(self.uplink.events),
            "aegis_restorations_count": self.restorer_aegis.total_restorations_performed,
            "aegis_alarm_steps": sum(1 for t in history if t.aegis_alarm),
            "scrubs_executed": self.scrub_controller.total_scrubs_executed,
            "aegis_recalibrate": self.aegis_recalibrate,
            "golden_int8": self.golden_int8,
            "orbit": self.orbit,
            "saa_minutes_total": round(sum(t.saa_minutes_in_step for t in history), 1),
            "golden_store_mb": {
                "aegis": round(self.store_aegis.total_compressed_bytes / 1e6, 2),
                "periodic_scrubbing": round(self.store_scrub.total_compressed_bytes / 1e6, 2),
            },
            "scrub_sweep": {
                rep["key"]: {
                    "mean_accuracy": round(sum(t.acc_scrub_sweep[rep["key"]] for t in history) / n_steps, 4),
                    "bytes_transferred": last.bytes_scrub_sweep[rep["key"]],
                    "energy_consumed_uj": round(rep["restorer"].cumulative_energy_uj, 2),
                }
                for rep in self.scrub_sweep
            },
        }
