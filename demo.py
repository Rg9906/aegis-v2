"""
Project Aegis-v2: Autonomous Demonstration Script for Reviewers.

Demonstrates:
1. Loading the pre-trained EuroSAT ResNet-18 satellite classifier.
2. Injecting an IEEE 754 exponent-MSB radiation bit-flip.
3. Live multi-tier health monitoring (detecting activation explosion and BN drift).
4. Detection-gated single-layer selective restore from the compressed golden store.
5. Multi-orbit comparative benchmark across all 5 strategies.
"""

import sys
import time
import torch

from aegis.config import EUROSAT_CLASSES
from aegis.data.eurosat import load_eurosat_splits
from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.eurosat_classifier import EuroSATResNet18
from aegis.monitor.orchestrator import AegisHealthMonitor
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.recalibrate import BNRecalibrator
from aegis.repair.selective_restore import DetectionGatedRestorer
from aegis.simulation.mission_sim import MissionSimulationHarness


def print_banner(text: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {text}")
    print("=" * 76)


def run_demonstration():
    print_banner("PROJECT AEGIS-v2: RADIATION-RESILIENT ONBOARD NEURAL NETWORKS")
    print("Mission: Sentinel-2 Land Cover Classification (EuroSAT 10-Class)")
    print("Payload: ResNet-18 Deep Convolutional Neural Network (11.18M Parameters)")

    # 1. Load Pretrained EuroSAT Model
    print("\n[Step 1/5] Initializing Pretrained EuroSAT ResNet-18 Model...")
    t0 = time.time()
    classifier = EuroSATResNet18(pretrained=True)
    registry = classifier.registry
    print(f"  [OK] Model loaded successfully in {time.time() - t0:.2f}s")
    print(f"  [OK] Layer blocks partitioned: {len(registry.block_ids)} restorable units")

    # Display block table
    print("\n  Partitioned Layer Blocks & Memory Footprint:")
    print(f"  {'Block ID':<18} {'Parameters':<12} {'FP32 Size':<12} {'INT8 Size'}")
    print("  " + "-" * 56)
    for b in registry.summary()[:6]:
        print(f"  {b['block_id']:<18} {b['params']:<12,} {b['fp32_mb']:.2f} MB      {b['int8_mb']:.2f} MB")
    print(f"  ... (total: {sum(b['params'] for b in registry.summary()):,} parameters)")

    # 2. Initialize Compressed Golden Store & Multi-Tier Monitor
    print("\n[Step 2/5] Initializing Onboard Compressed Golden Store & Multi-Tier Monitor...")
    store = CompressedGoldenStore(registry)
    print(f"  [OK] Uncompressed Model Size: {store.total_uncompressed_bytes / (1024*1024):.2f} MB")
    print(f"  [OK] Compressed Golden Store: {store.total_compressed_bytes / (1024*1024):.2f} MB (zstd level 3)")
    print(f"  [OK] Onboard Compression Ratio: {store.overall_compression_ratio:.2f}x")

    # Real Sentinel-2 EuroSAT chips: disjoint unlabeled calibration set + labeled evaluation set
    torch.manual_seed(42)
    splits = load_eurosat_splits(calib_per_class=10, eval_per_class=10, seed=0)
    val_images, val_labels = splits.eval_images, splits.eval_labels
    print(f"  [OK] Loaded {splits.calib_images.size(0)} calibration + {val_labels.numel()} evaluation EuroSAT chips")

    monitor = AegisHealthMonitor(classifier.backbone, registry)
    monitor.calibrate(splits.calib_images)
    monitor.attach()
    print("  [OK] Multi-Tier Health Monitor calibrated and attached to forward hooks.")

    recalibrator = BNRecalibrator(buffer_capacity=64)
    recalibrator.ingest_unlabeled_imagery(splits.calib_images)
    restorer = DetectionGatedRestorer(registry, store, recalibrator)

    # 3. Baseline Clean Inference
    preds_clean, _ = classifier.predict(val_images[:4])
    clean_acc = classifier.evaluate(val_images, val_labels)["accuracy"]
    print("\n[Step 3/5] Error-Free Baseline Inference:")
    for i, p in enumerate(preds_clean.tolist()):
        truth = EUROSAT_CLASSES[val_labels[i].item()]
        print(f"  Sample {i+1}: predicted {EUROSAT_CLASSES[p]:<22} (truth: {truth})")
    print(f"  Clean accuracy on {val_labels.numel()} chips: {clean_acc * 100:.1f}%")

    report_clean = monitor.diagnose_current_state()
    print(f"  Monitor Diagnostic: is_healthy={report_clean.is_healthy}, flagged_blocks={report_clean.flagged_blocks}")

    # 4. Inject Realistic IEEE 754 Exponent-MSB Bit-Flip
    print_banner("SIMULATING PHYSICAL RADIATION EVENT (LEO SEU STRIKE)")
    target_block = "layer3_block1"
    print(f"  Injecting Single Event Upset (SEU) into [{target_block}]...")
    injector = ModelFaultInjector(classifier.backbone, registry)
    rec = injector.inject_targeted_bit_flip(block_id=target_block, bit_position=FLOAT32_EXPONENT_MSB)

    print(f"  Target Parameter: {rec.param_name} [index {rec.flat_index}]")
    print(f"  Pre-Flip Value  : {rec.pre_value:.6f}")
    print(f"  Post-Flip Value : {rec.post_value:.6e}  <-- EXPLOSIVE EXPONENT-MSB ACTIVATION SPIKE!")

    # Corrupted forward pass
    monitor.clear()
    preds_corrupt, _ = classifier.predict(val_images[:16])
    report_corrupt = monitor.diagnose_current_state()
    monitor.clear()
    corrupt_acc = classifier.evaluate(val_images, val_labels)["accuracy"]
    print(f"  Corrupted accuracy: {corrupt_acc * 100:.1f}%  (clean: {clean_acc * 100:.1f}%)")

    print(f"\n  Post-Radiation Health Diagnosis:")
    print(f"  [ALERT] is_healthy: {report_corrupt.is_healthy}")
    print(f"  [ALERT] Flagged Blocks: {report_corrupt.flagged_blocks}")
    print(f"  [ALERT] Catastrophic Anomaly: {report_corrupt.has_catastrophic_anomaly}")

    # 5. Detection-Gated Layer-Selective Restore
    print_banner("AEGIS-v2 AUTONOMOUS IN-SITU REPAIR")
    print(f"  Gating repair: Restoring ONLY CRC-confirmed block(s) {report_corrupt.tier3_corrupted_blocks}...")
    t_repair = time.perf_counter()
    # Lossless golden store -> bit-exact restore, no BN recalibration needed
    event = restorer.process_health_report(report_corrupt, auto_recalibrate=False)
    repair_duration = (time.perf_counter() - t_repair) * 1000.0

    print(f"  [REPAIRED] Restored Blocks       : {event.restored_blocks}")
    print(f"  [REPAIRED] Data Transferred      : {event.total_bytes_transferred / 1024.0:.2f} KB (vs {store.total_uncompressed_bytes / (1024*1024):.2f} MB full reload)")
    print(f"  [REPAIRED] Energy Consumed       : {event.total_energy_uj:.2f} uJ")
    print(f"  [REPAIRED] Restoration Downtime  : {repair_duration:.2f} ms")
    if event.recalibration_report:
        print(f"  [REPAIRED] Label-Free Recalib    : Processed {event.recalibration_report.samples_processed} buffered images")

    # Post-restore forward pass
    monitor.clear()
    preds_restored, _ = classifier.predict(val_images[:4])
    report_restored = monitor.diagnose_current_state(run_tier3_scan=True)
    monitor.clear()
    restored_acc = classifier.evaluate(val_images, val_labels)["accuracy"]
    print(f"  [OK] Post-Repair Health    : is_healthy={report_restored.is_healthy} (full CRC scan)")
    print(f"  [OK] Prediction Matches Clean: {torch.equal(preds_clean, preds_restored)}")
    print(f"  [OK] Post-Repair Accuracy  : {restored_acc * 100:.1f}%")

    # 6. Mini Mission Comparative Campaign
    print_banner("MULTI-ORBIT COMPARATIVE CAMPAIGN (AEGIS vs BASELINES)")
    print("  Simulating 24-hour LEO mission (6 SAA crossings, 2-hour steps) on real EuroSAT chips...")
    monitor.detach()
    sim = MissionSimulationHarness(
        base_model=classifier.backbone,
        val_images=val_images,
        val_labels=val_labels,
        calib_images=splits.calib_images,
        random_seed=42,
    )
    sim.run_simulation(total_days=1.0, step_minutes=120.0)
    summary = sim.get_campaign_summary()

    print("\n  Campaign Summary Results (24 Simulated LEO Hours):")
    print(f"  {'Strategy':<26} {'Mean Accuracy':<16} {'Energy Consumed':<18} {'Bytes Transferred'}")
    print("  " + "-" * 72)
    accs = summary["mean_accuracy"]
    energies = summary["energy_consumed_uj"]
    bytes_tf = summary["bytes_transferred"]

    print(f"  {'Unmitigated Baseline':<26} {accs['unmitigated']*100:.1f}%             {'N/A':<18} {'0 KB'}")
    print(f"  {'Ranger Activation Clamping':<26} {accs['ranger_clipping']*100:.1f}%             {'N/A (Compute)':<18} {'0 KB'}")
    print(f"  {'BLINK In-Situ Re-Centering':<26} {accs['blink_recentering']*100:.1f}%             {'N/A (Compute)':<18} {'0 KB'}")
    print(f"  {'Periodic Blind Scrubbing':<26} {accs['periodic_scrubbing']*100:.1f}%             {energies['periodic_scrubbing']:.1f} uJ          {bytes_tf['periodic_scrubbing']/1024:.1f} KB")
    print(f"  {'Aegis-v2 Selective Restore':<26} {accs['aegis_v2']*100:.1f}%             {energies['aegis_v2']:.1f} uJ          {bytes_tf['aegis_v2']/1024:.1f} KB")
    print("  " + "-" * 72)
    print(f"  >>> Aegis-v2 Energy Savings vs Periodic Scrubbing: {energies['energy_savings_percent']:.1f}%")
    print(f"  >>> Autonomous Restorations Executed: {summary['aegis_restorations_count']}")
    print(f"  >>> Selective Restore Efficiency: {bytes_tf['aegis_v2']/max(bytes_tf['periodic_scrubbing'],1)*100:.1f}% of full-scrub bandwidth")

    print_banner("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    run_demonstration()
