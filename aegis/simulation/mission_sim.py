"""
LEO Orbital Mission Simulation and Benchmark Campaign Substructure.

Executes time-stepped multi-orbit simulations modeling accumulated radiation flux,
SAA passes, and comparative resilience evaluation across all mitigation strategies.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import copy
import torch
import torch.nn as nn

from aegis.baselines.blink_recentering import BlinkRecenteringMitigator
from aegis.baselines.periodic_scrubbing import PeriodicScrubbingController
from aegis.baselines.ranger_clipper import RangerActivationClipper
from aegis.config import DEFAULT_RAD_CONFIG, RadiationEnvironmentConfig
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.fault_engine.radiation_model import LEORadiationEnvironment, OrbitalState
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
    sbus_injected_step: int
    mbus_injected_step: int
    cumulative_faults: int
    acc_unmitigated: float
    acc_ranger: float
    acc_blink: float
    acc_periodic_scrub: float
    acc_aegis: float
    energy_aegis_uj: float
    energy_scrub_uj: float
    bytes_aegis: int
    bytes_scrub: int
    aegis_restored_blocks: List[str]


class MissionSimulationHarness:
    """
    Coordinates side-by-side execution of all mitigation approaches
    under identical, physically grounded LEO radiation flux.
    """

    def __init__(
        self,
        base_model: nn.Module,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
        rad_config: RadiationEnvironmentConfig = DEFAULT_RAD_CONFIG,
        random_seed: int = 42,
    ):
        self.val_images = val_images
        self.val_labels = val_labels
        self.rad_config = rad_config

        # 1. Models for each strategy (cloned independently)
        self.model_unmitigated = copy.deepcopy(base_model)
        self.model_ranger = copy.deepcopy(base_model)
        self.model_blink = copy.deepcopy(base_model)
        self.model_scrub = copy.deepcopy(base_model)
        self.model_aegis = copy.deepcopy(base_model)

        # 2. Registries
        self.reg_unmitigated = LayerRegistry(self.model_unmitigated)
        self.reg_ranger = LayerRegistry(self.model_ranger)
        self.reg_blink = LayerRegistry(self.model_blink)
        self.reg_scrub = LayerRegistry(self.model_scrub)
        self.reg_aegis = LayerRegistry(self.model_aegis)

        # 3. Fault Injectors for each model
        self.inj_unmitigated = ModelFaultInjector(self.model_unmitigated, self.reg_unmitigated)
        self.inj_ranger = ModelFaultInjector(self.model_ranger, self.reg_ranger)
        self.inj_blink = ModelFaultInjector(self.model_blink, self.reg_blink)
        self.inj_scrub = ModelFaultInjector(self.model_scrub, self.reg_scrub)
        self.inj_aegis = ModelFaultInjector(self.model_aegis, self.reg_aegis)

        # 4. Mitigation Engines
        self.rad_env = LEORadiationEnvironment(rad_config, random_seed=random_seed)

        # Ranger Setup
        self.ranger_clipper = RangerActivationClipper()
        self.ranger_clipper.calibrate(self.model_ranger, val_images[:8])
        self.ranger_clipper.attach(self.model_ranger)

        # BLINK Setup
        self.blink_mitigator = BlinkRecenteringMitigator(drift_threshold=3.0)
        self.blink_mitigator.attach(self.model_blink)

        # Periodic Scrub Setup
        self.store_scrub = CompressedGoldenStore(self.reg_scrub)
        self.restorer_scrub = DetectionGatedRestorer(self.reg_scrub, self.store_scrub)
        self.scrub_controller = PeriodicScrubbingController(self.restorer_scrub, scrub_interval_hours=12.0)

        # Aegis-v2 Setup
        self.store_aegis = CompressedGoldenStore(self.reg_aegis)
        self.recal_aegis = BNRecalibrator(buffer_capacity=64)
        self.recal_aegis.ingest_unlabeled_imagery(val_images[:32])
        self.restorer_aegis = DetectionGatedRestorer(self.reg_aegis, self.store_aegis, self.recal_aegis)
        self.monitor_aegis = AegisHealthMonitor(self.model_aegis, self.reg_aegis)
        self.monitor_aegis.calibrate(val_images[:8])
        self.monitor_aegis.attach()

        self.telemetry_history: List[TimeStepTelemetry] = []

    def _evaluate_model(self, model: nn.Module) -> float:
        """Computes top-1 classification accuracy on validation set."""
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(self.val_images.to(device))
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == self.val_labels.to(device)).sum().item()
            total = self.val_labels.numel()
            return float(correct / total if total > 0 else 0.0)

    def run_simulation(self, total_days: float = 3.0, step_minutes: float = 60.0) -> List[TimeStepTelemetry]:
        """
        Executes multi-day orbital campaign, logging telemetry across all strategies.
        """
        total_steps = int((total_days * 1440.0) // step_minutes)
        total_model_bits = sum(b.total_bits_fp32 for b in self.reg_aegis._blocks.values())

        for step_idx in range(total_steps):
            # 1. Advance orbital clock
            orbital_state = self.rad_env.step(delta_minutes=step_minutes, total_model_bits=total_model_bits)

            # 2. Inject identical faults across all 5 models
            # Use "exponent" field targeting: these are the physically destructive upsets
            # that cause activation explosions and are reliably detected by T1 behavioral monitor
            records = self.inj_unmitigated.inject_orbital_flux_step(orbital_state, target_field="exponent")
            for rec in records:
                # Mirror same bit-flip on all other models for fair comparison
                self.inj_ranger.inject_targeted_bit_flip(rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)
                self.inj_blink.inject_targeted_bit_flip(rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)
                self.inj_scrub.inject_targeted_bit_flip(rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)
                self.inj_aegis.inject_targeted_bit_flip(rec.block_id, rec.param_name, rec.flat_index, rec.bit_position)

            # 3. Execute Periodic Scrubbing baseline step
            scrub_event = self.scrub_controller.step(self.rad_env.elapsed_minutes)

            # 4. Execute Aegis-v2 monitoring and detection-gated selective repair
            self.monitor_aegis.clear()
            # Forward pass on evaluation batch triggers runtime hooks
            _ = self._evaluate_model(self.model_aegis)
            # run_tier3_scan=False: T3 checksum scan is only triggered when T1/T2 flag anomalies
            # This is the correct Aegis-v2 architecture: behavior-gated confirmatory checksums
            report = self.monitor_aegis.diagnose_current_state(run_tier3_scan=False)

            aegis_event = None
            if not report.is_healthy:
                aegis_event = self.restorer_aegis.process_health_report(report, auto_recalibrate=True)
                # Refresh T3 golden checksums to reflect restored + recalibrated BN stats
                self.monitor_aegis.refresh_golden_checksums()

            # 5. Evaluate accuracies of all models
            acc_unmitigated = self._evaluate_model(self.model_unmitigated)
            acc_ranger = self._evaluate_model(self.model_ranger)
            acc_blink = self._evaluate_model(self.model_blink)
            acc_scrub = self._evaluate_model(self.model_scrub)
            acc_aegis = self._evaluate_model(self.model_aegis)

            # 6. Record telemetry
            step_record = TimeStepTelemetry(
                step_index=step_idx + 1,
                elapsed_minutes=round(self.rad_env.elapsed_minutes, 1),
                elapsed_days=round(self.rad_env.elapsed_days, 2),
                orbit_index=self.rad_env.current_orbit_index,
                in_saa=orbital_state.in_saa,
                sbus_injected_step=orbital_state.total_sbus_generated,
                mbus_injected_step=orbital_state.total_mbus_generated,
                cumulative_faults=self.inj_unmitigated.total_faults_injected,
                acc_unmitigated=round(acc_unmitigated, 4),
                acc_ranger=round(acc_ranger, 4),
                acc_blink=round(acc_blink, 4),
                acc_periodic_scrub=round(acc_scrub, 4),
                acc_aegis=round(acc_aegis, 4),
                energy_aegis_uj=round(self.store_aegis._cumulative_energy_uj, 2),
                energy_scrub_uj=round(self.store_scrub._cumulative_energy_uj, 2),
                bytes_aegis=self.store_aegis._cumulative_bytes_read,
                bytes_scrub=self.store_scrub._cumulative_bytes_read,
                aegis_restored_blocks=aegis_event.restored_blocks if aegis_event else [],
            )
            self.telemetry_history.append(step_record)

        return self.telemetry_history

    def get_campaign_summary(self) -> Dict[str, object]:
        """Calculates area-under-accuracy, total energy, and reliability metrics."""
        if not self.telemetry_history:
            return {}

        n_steps = len(self.telemetry_history)
        mean_acc_unmit = sum(t.acc_unmitigated for t in self.telemetry_history) / n_steps
        mean_acc_ranger = sum(t.acc_ranger for t in self.telemetry_history) / n_steps
        mean_acc_blink = sum(t.acc_blink for t in self.telemetry_history) / n_steps
        mean_acc_scrub = sum(t.acc_periodic_scrub for t in self.telemetry_history) / n_steps
        mean_acc_aegis = sum(t.acc_aegis for t in self.telemetry_history) / n_steps

        last = self.telemetry_history[-1]
        energy_savings_pct = (
            (last.energy_scrub_uj - last.energy_aegis_uj) / last.energy_scrub_uj * 100.0
            if last.energy_scrub_uj > 0
            else 0.0
        )

        return {
            "total_simulated_days": last.elapsed_days,
            "total_faults_accumulated": last.cumulative_faults,
            "mean_accuracy": {
                "unmitigated": round(mean_acc_unmit, 4),
                "ranger_clipping": round(mean_acc_ranger, 4),
                "blink_recentering": round(mean_acc_blink, 4),
                "periodic_scrubbing": round(mean_acc_scrub, 4),
                "aegis_v2": round(mean_acc_aegis, 4),
            },
            "energy_consumed_uj": {
                "periodic_scrubbing": last.energy_scrub_uj,
                "aegis_v2": last.energy_aegis_uj,
                "energy_savings_percent": round(energy_savings_pct, 2),
            },
            "bytes_transferred": {
                "periodic_scrubbing": last.bytes_scrub,
                "aegis_v2": last.bytes_aegis,
            },
            "aegis_restorations_count": self.restorer_aegis.total_restorations_performed,
        }
