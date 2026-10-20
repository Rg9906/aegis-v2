# Project Aegis-v2: Research Study, Open-Source Landscape & Fast Prototype Blueprint

**Date:** October 2026  
**Objective:** Deliver an in-depth research study, identify reusable open-source modules and papers, and establish an actionable architecture to build a working prototype by the end of the week.

---

## 1. Project Understanding & The Core Problem

### The Starting Point (Aegis-v1)
The project initially envisioned a mechanism combining:
1. Dead-neuron / low-firing monitoring,
2. Weight-zeroing fault models, and
3. Noise-shock / micro-reset "neuron revival" (borrowed from RL plasticity concepts like ReDo) on a toy Multi-Layer Perceptron (MLP).

### Why Aegis-v1 is Unpublishable & Rejected by Reviewers
1. **The Fault-Model Mismatch (Physics of Radiation Upsets):**
   - Single Event Upsets (SEUs) in Low Earth Orbit (LEO) strike memory bits at random.
   - In single-precision IEEE 754 (`float32`), flipping the most significant bit (MSB) of the 8-bit exponent (bit 30) converts a nominal trained weight (e.g., $0.75$) into an astronomical value:
     $$0.75 \longrightarrow 2.55 \times 10^{38}$$
   - This creates explosive activations ($\infty / \text{NaN}$) throughout downstream layers. It does **not** create dead or zeroed neurons. Any reviewer in hardware reliability or space computing will immediately reject a paper testing zeroed weights or dead-neuron monitors for radiation resilience.
2. **The Plasticity Fallacy in Inference-Only Orbit:**
   - Methods like ReDo or Continual Backprop rely on an active learning loop with gradient updates.
   - In an operational Earth Observation (EO) satellite payload (e.g. Sentinel-2 land-use classification), there is **no ongoing backpropagation or ground truth**. Injecting random noise into corrupted weights merely randomizes the capacity—it cannot reconstruct the specific learned visual mapping.
3. **Network Architecture Gap:**
   - Flat MLPs do not model modern spaceborne vision workloads. Reviewers expect Convolutional Neural Networks with Batch Normalization (e.g., ResNet-18, MobileNetV2) or lightweight Vision Transformers (ViT-Tiny).

---

## 2. The Defensible, Publishable Reframing (Aegis-v2)

> **"Detection-Gated, Layer-Selective Restore for Onboard Earth Observation Classifiers under Accumulated LEO Radiation Faults and an Energy/Bandwidth Budget."**

### The 4-Pillar Architecture
1. **Multi-Tier In-Situ Health Monitor (Label-Free):**
   - **Tier 1 (Activation Magnitude Envelope / Ranger):** Catches exponent-MSB flips and NaN anomalies in $O(1)$ time without compute overhead.
   - **Tier 2 (Batch Normalization Drift / BLINK):** Measures statistical distance ($Z$-score / Wasserstein) between runtime batch statistics $(\mu_B, \sigma_B^2)$ and calibrated clean BN statistics $(\mu_0, \sigma_0^2)$. Catches subtle mantissa and lower-exponent corruptions.
   - **Tier 3 (Lightweight Checksums / RADAR):** Fast CRC-32 checksum verification per layer block.
2. **Detection-Gated Layer-Selective Restore:**
   - Instead of costly periodic blind scrubbing (which reloads the entire $44.7\text{ MB}$ model and wastes energy) or slow uplink reloads from Earth ($\approx 1\text{ Mbps}$ uplink on satellites like Chaohu-1), Aegis-v2 reloads **only the corrupted layer** ($\sim 50\text{ KB to } 1.5\text{ MB}$) from an onboard compressed golden copy.
3. **Compressed Rad-Tolerant Golden Store:**
   - Model weights stored in a rad-hard companion SPI flash or MRAM, compressed using INT8 quantization and `zstandard` (`zstd`), taking only $\approx 2.9\text{ MB}$ total.
4. **Label-Free Post-Repair Recalibration:**
   - After hot-patching the damaged layer, incoming unlabeled satellite image chips pass through the network to update downstream Batch Normalization statistics before resuming classification.
5. **LEO Orbit- & SAA-Aware Radiation Model:**
   - Real-world orbit physics: $\sim 73.4\%$ of all upsets occur in the South Atlantic Anomaly (SAA) (based on Flying Laptop and Alsat-1 measurements).

---

## 3. Open-Source Landscape & Substructure Reuse Matrix

| Substructure Needed | Best Open-Source Tools & Repos | Status & How We Leverage It |
| :--- | :--- | :--- |
| **LEO Fault Injector** | • **PyTorchFI** (UBC)<br>• **IntelLabs PyTorchALFI**<br>• **langekevin/mlspace_robustness** (IEEE SCC 2024) | • We implement a high-speed, vectorized **IEEE 754 bit-flip engine** via PyTorch tensor reinterpretation (`tensor.view(torch.int32) ^= (1 << bit)`).<br>• Incorporates LEO upset rates ($10^{-8}$ to $10^{-6}$ upsets/bit/day), 20% clustered MBUs, and SAA passage multipliers. |
| **EO Satellite Target Model** | • **EuroSAT** (Sentinel-2 10-class dataset)<br>• Pre-trained weights on Hugging Face (`cm93/resnet18-eurosat`) | • **Verified & downloaded:** Standard `torchvision.models.resnet18` with EuroSAT 10-class weights (~97.7% clean accuracy). Ready out-of-the-box. |
| **In-Situ Health Monitor** | • **BLINK** (arXiv:2609.06781, Sept 2026)<br>• **Ranger** (arXiv:2003.13874, DSN 2021)<br>• **RADAR** (DATE 2021) | • Combine Ranger activation bounds (instant NaN/explosion flagging) with BLINK BN running statistics drift tracking. |
| **Comparative Baselines** | • Unmitigated Baseline<br>• Ranger Activation Clipping<br>• BLINK Activation Re-centering<br>• Periodic Full Scrubbing<br>• Ground Uplink Reload (Chaohu-1 model) | • Implement all 5 baselines to demonstrate Aegis-v2's Pareto dominance in **Accuracy vs. Energy & Bandwidth**. |
| **Compression & Storage** | • `safetensors` (v0.8.0)<br>• `zstandard` (v0.25.0) | • Both modules are already installed in Python 3.13. Enables sub-millisecond single-layer decompression and injection. |
| **Reviewer Dashboard** | • `streamlit` (v1.55.0)<br>• `matplotlib` / `plotly` | • Interactive live mission telemetry dashboard: shows orbit path, SAA crossing, live layer health heatmap, accuracy-over-time curves, and Sentinel-2 classification confusion matrix. |

---

## 4. End-of-Week Prototype Implementation Plan

```
[Day 1] Pre-trained Model & IEEE 754 Fault Engine (Completed / Verified)
   └── EuroSAT ResNet-18 loaded; bit-flip tensor view tested; exponent-MSB explosion validated.

[Day 2] In-Situ Detection Engine & Mitigation Baselines
   └── Build Ranger bounds tracker, BLINK BN-drift detector, and Ranger/BLINK baselines.

[Day 3] Compressed Golden Store & Layer-Selective Restore
   └── Build zstd safetensors layer cache, hot-patching pipeline, and label-free recalibrator.

[Day 4] Orbit Simulation & Comparative Benchmark Campaign
   └── Simulate 30 LEO orbit days with SAA passages; compute Accuracy-vs-Time & Energy/Byte Pareto curves.

[Day 5] Streamlit Mission Console & Reviewer Demo Rehearsal
   └── Finalize interactive dashboard, generate publication-grade figures, and rehearse reviewer pitch.
```

---

## 5. Next Steps for Immediate Execution

1. Build `aegis/` package directory structure with the modular engine files.
2. Generate the **Exponent-MSB vs. Dead-Neuron demonstration script** to provide undeniable visual proof of the radiation fault model for reviewers.
3. Integrate the EuroSAT validation pipeline with the live selective-restore harness.
