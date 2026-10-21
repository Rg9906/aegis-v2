"""
Unit tests for Aegis Baselines, EO Confusion Analysis, and Mission Simulation.
"""

import pytest
import torch
import torchvision.models as models
import numpy as np

from aegis.baselines.blink_recentering import BlinkRecenteringMitigator
from aegis.baselines.periodic_scrubbing import PeriodicScrubbingController
from aegis.baselines.ranger_clipper import RangerActivationClipper
from aegis.config import EUROSAT_NUM_CLASSES
from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.layer_registry import LayerRegistry
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.selective_restore import DetectionGatedRestorer
from aegis.simulation.confusion_analyzer import EOConfusionAnalyzer
from aegis.simulation.mission_sim import MissionSimulationHarness


@pytest.fixture
def base_test_env():
    torch.manual_seed(42)
    model = models.resnet18(weights=None, num_classes=EUROSAT_NUM_CLASSES)
    val_images = torch.randn(8, 3, 64, 64)
    val_labels = torch.randint(0, EUROSAT_NUM_CLASSES, (8,))
    return model, val_images, val_labels


def test_ranger_clipper(base_test_env):
    """Verifies that Ranger clamps activations and prevents NaN/Inf propagation."""
    model, val_images, _ = base_test_env
    clipper = RangerActivationClipper()
    clipper.calibrate(model, val_images)
    clipper.attach(model)

    assert len(clipper.bounds) > 0

    with torch.no_grad():
        out = model(val_images)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    clipper.detach()


def test_blink_recentering(base_test_env):
    """Verifies that BLINK in-situ re-centering hooks into BN layers."""
    model, val_images, _ = base_test_env
    blink = BlinkRecenteringMitigator(drift_threshold=2.0)
    blink.attach(model)

    with torch.no_grad():
        out = model(val_images)
    assert not torch.isnan(out).any()

    blink.detach()


def test_periodic_scrubbing(base_test_env):
    """Verifies timer logic for blind periodic full scrubbing."""
    model, _, _ = base_test_env
    registry = LayerRegistry(model)
    store = CompressedGoldenStore(registry)
    restorer = DetectionGatedRestorer(registry, store)
    controller = PeriodicScrubbingController(restorer, scrub_interval_hours=12.0)

    # Minute 100: Should not trigger (< 720 minutes)
    event1 = controller.step(100.0)
    assert event1 is None
    assert controller.total_scrubs_executed == 0

    # Minute 720: Should trigger first scrub (12 hours elapsed)
    event2 = controller.step(720.0)
    assert event2 is not None
    assert controller.total_scrubs_executed == 1

    # Minute 1000: Should not trigger (< 1440 minutes)
    assert controller.step(1000.0) is None
    assert controller.total_scrubs_executed == 1

    # Minute 1440: Should trigger second scrub (24 hours elapsed)
    event3 = controller.step(1440.0)
    assert event3 is not None
    assert controller.total_scrubs_executed == 2


def test_eo_confusion_analyzer(base_test_env):
    """Verifies confusion matrix generation and delta reporting."""
    model, val_images, val_labels = base_test_env
    analyzer = EOConfusionAnalyzer()

    clean_cm, clean_acc = analyzer.compute_confusion_matrix(model, val_images, val_labels)
    assert clean_cm.shape == (10, 10)
    assert clean_cm.sum() == val_labels.numel()

    # Corrupt model
    registry = LayerRegistry(model)
    injector = ModelFaultInjector(model, registry)
    injector.inject_targeted_bit_flip("layer4_block1", bit_position=FLOAT32_EXPONENT_MSB)

    corrupt_cm, corrupt_acc = analyzer.compute_confusion_matrix(model, val_images, val_labels)
    delta_report = analyzer.analyze_confusion_delta(clean_cm, clean_acc, corrupt_cm, corrupt_acc)

    assert delta_report.confusion_matrix_clean.shape == (10, 10)
    assert delta_report.confusion_matrix_corrupted.shape == (10, 10)


def test_mission_simulation_harness_mini_run(base_test_env):
    """Verifies end-to-end multi-orbit campaign simulation across all 5 strategies."""
    model, val_images, val_labels = base_test_env

    sim = MissionSimulationHarness(
        base_model=model,
        val_images=val_images,
        val_labels=val_labels,
        random_seed=42,
    )

    # Run a short simulation of 3 steps (step_minutes=60, total_days=0.125 = 3 hours)
    telemetry = sim.run_simulation(total_days=0.125, step_minutes=60.0)
    assert len(telemetry) == 3

    summary = sim.get_campaign_summary()
    assert "mean_accuracy" in summary
    assert "energy_consumed_uj" in summary
    assert "aegis_v2" in summary["mean_accuracy"]
    assert "periodic_scrubbing" in summary["mean_accuracy"]


def test_scrub_sweep_and_pareto_points(base_test_env):
    """Extra scrubbing replicas share the fault stream and feed the accuracy-vs-bandwidth Pareto."""
    from aegis.reporting import pareto_points

    model, val_images, val_labels = base_test_env
    sim = MissionSimulationHarness(model, val_images, val_labels, random_seed=1, scrub_sweep_hours=(1.0,))
    telemetry = sim.run_simulation(total_days=0.125, step_minutes=60.0)
    assert set(telemetry[-1].acc_scrub_sweep) == {"1"}
    # 1-hour scrub fires every step -> 3 full reloads
    assert sim.scrub_sweep[0]["controller"].total_scrubs_executed == 3

    summary = sim.get_campaign_summary()
    summary["seed"] = 1
    from dataclasses import asdict
    points = pareto_points({1: {"summary": summary, "telemetry": [asdict(t) for t in telemetry]}})
    strategies = {p["strategy"] for p in points}
    assert "Periodic scrubbing (1 h)" in strategies and "Aegis-v2 selective restore" in strategies
