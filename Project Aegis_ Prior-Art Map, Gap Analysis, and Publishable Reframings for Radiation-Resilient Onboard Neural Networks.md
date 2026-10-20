# Project Aegis: Prior-Art Map, Gap Analysis, and Publishable Reframings for Radiation-Resilient Onboard Neural Networks

Aegis as currently framed (dead-neuron monitoring + noise/reset "revival" on an MLP with zeroed-weight faults) is not publishable: every component already exists, and its fault model does not match how radiation actually damages float networks. A narrower, defensible gap does remain, though. As far as we could find, no peer-reviewed work combines (a) an activation/BN-statistics health monitor, (b) layer-selective reload from a compressed protected copy, (c) label-free post-repair recalibration, and (d) scheduling of that repair by energy budget or orbital radiation exposure (e.g. South Atlantic Anomaly passes). The closest competitor, BLINK (arXiv 2609.06781, September 2026), appeared only weeks ago and must be the student's primary baseline.

## TL;DR

- **The original idea is covered.** Range restriction (Ranger, FT-ClipAct, ProAct), weight correction (MILR, CoG, weight reconstruction), checksum detection (RADAR), BN-based in-situ detection/mitigation (BLINK), application-aware radiation tolerance validated on a real satellite payload (RedNet/Chaohu-1), and multi-bit EDAC for onboard models (ZEDAC) together cover detection, mitigation and recovery. The "neuron revival" tier is essentially ReDo, a training-time plasticity method that cannot restore learned function after hardware faults.
- **The fault model is the biggest technical weakness.** Real SEUs in float32 weights produce enormous values (an exponent-MSB flip turns 0.75 into ~2.55×10^38), not dead neurons. Beam experiments show single-bit-flip injection can underestimate real misdetection rates by 8.66×. A zero-weight, dead-neuron simulator will be rejected by any reliability reviewer.
- **The publishable reframing:** "detection-gated, layer-selective restore for onboard EO classifiers under accumulated LEO faults and an energy/bandwidth budget". Evaluate it on EuroSAT with PyTorchFI against ECC/scrubbing, Ranger, BLINK, and full reload. The safest first paper is the benchmark/empirical study (Reframing A); the higher-novelty follow-up is the scheduler (Reframing B).

## Key Findings

1. **Hardware mitigation is mature and is the default.** ECC/EDAC, TMR and scrubbing are standard. Alsat-1 measurements showed that a double-bit EDAC "scheme is sufficient to handle the typical SEU rate at this LEO environment".\[1\] COTS AI payloads are typically flown behind a rad-tolerant supervisor that watchdogs and power-cycles them.\[2\]
2. **Software DNN fault tolerance is crowded.** Ranger reports 3×–50× resilience improvement with no accuracy loss;\[3\] FT-ClipAct reports a 68.92% average accuracy improvement for VGG-16 at a 1e-5 fault rate;\[4\] CoG-guided correction reports up to 49.55× (ResNet-18) at BER 1e-3.\[5\] BLINK recovers ResNet-50/CIFAR-10 from 10% to 85.88% under 0.5% random bit flips with <2% latency overhead, with no labels and no host.\[6\]
3. **Space-specific DNN work exists and is recent.** RedNet was validated on the Chaohu-1 SAR satellite's Jetson Xavier NX payload.\[7\] The Lange et al. survey (IEEE SCC 2024) found prior work "did not thoroughly examine the impact of natural hazards on ML models meant for spacecraft".\[8\] ZEDAC (SmallSat 2026) restores corrupted model parameters "without retraining or ground intervention".\[9\]
4. **"Self-healing" papers mostly rely on training data, golden data, or algebraic redundancy.** Examples: SHNN in Scientific Reports 2026 retrains the afflicted layer;\[10\] MILR recomputes weights from stored layer input/output relationships.\[11\] None addresses the onboard constraint of no labels, limited energy, and intermittent uplink in a radiation-calibrated way.
5. **Repair scheduling has not been done for DNNs.** We found adaptive scrubbing for FPGA configuration memory (DAC 2015) and position papers calling for "radiation-aware" AI runtimes (arXiv 2608.21034).\[12\]\[13\] We found no peer-reviewed paper that schedules DNN-weight scrubbing or reload by energy budget or by SAA crossing.

## Details

### 1. Map of existing solutions by category

**Hardware level (rad-hard, ECC, EDAC, TMR, scrubbing)**
- **SECDED ECC adapted to DNNs.** arXiv 2508.12347 corrects single-bit errors and zeroes a weight on detected double errors, reporting >300% accuracy gain at BER 10^-1 for 47.5% area overhead.\[14\]
- **Selective TMR and selective hardening for FPGAs.** Libano et al., IEEE TNS 2018; Bertoa et al., IEEE Design & Test.\[15\]
- **Safe-NEureka (arXiv 2602.04803).** A hybrid DMR/performance-mode RISC-V DNN accelerator for satellite AI, with ECC memory and a TMR controller in GF 12nm.\[16\]
- **ZEDAC (SmallSat 2026).** Universal multi-bit EDAC for model parameters across DRAM/SRAM and SD/eMMC storage, via a rad-tolerant SPI companion. Recovery effectiveness depends on the corruption pattern (random vs burst, density, locality).\[9\]
- **Caveat from beam testing.** Fernandes dos Santos et al. (IEEE TNS 2023, DOI 10.1109/TNS.2022.3224538; arXiv 2211.13094) found up to 61% of DNN errors on GPUs were critical and that ECC was ineffective in reducing critical errors.\[17\]\[18\] The reason is that many faults strike logic, not protected memory.

**Algorithm-based fault tolerance (ABFT) and checksums**
- **ABFT for convolutions.** FT-CNN (Zhao et al., IEEE TPDS); Hari et al., "Making Convolutions Resilient via Algorithm-Based Error Detection" (IEEE TDSC).\[5\]
- **Checksums over embedded CNNs.** Liu et al., Microelectronics Reliability 2022, uses per-channel weight checksums plus per-layer output checksums: ~100% coverage at 4% time and 17% memory overhead.\[5\]
- **TR-ABFT for spaceborne NPUs** (from your list; not re-verified by us).

**Range restriction / clipping (software, no retraining)**
- Ranger (arXiv 2003.13874; DSN 2021) and FT-ClipAct (DATE 2020).\[3\]\[19\]
- ProAct (arXiv 2406.06313), with per-neuron hybrid clipped ReLU.\[20\]
- Global Clipper for transformers (arXiv 2406.03229) and activation range supervision (arXiv 2108.07019).\[19\]
- Limitation of RedNet-style clipping: SAVE (USENIX ATC 2025) reports that RedNet "experiences significantly faster accuracy degradation under physically contiguous flips" and becomes almost ineffective after 1,500 flips.\[21\]

**Weight hardening, quantization, fault-aware training**
- **Weight transforms.** Weight scaling (arXiv 2411.18993) and saturated weight transforms (arXiv 2411.19027). A public review of the latter flags an under-specified BER protocol and unhandled NaN cases.\[5\]\[22\]
- **Fault-aware quantization.** FAQ (arXiv 2305.12590) shows that protecting only the first and last layers "can result in a significant improvement in the DNN accuracy".\[23\]
- **Fault-aware training.** FAT-RABBIT (fault-aware training against bit-flip attacks).\[24\]

**Selective / layer-aware protection and vulnerability analysis**
- **DeepVigor (arXiv 2303.06931) and DeepVigor+.** Analytical vulnerability value ranges per neuron and per bit, reported 99.9–100% accurate.\[25\]\[26\]\[27\]
- **SPINE (arXiv 2606.19526).** A GDB-driven profiler for quantized networks under accumulated weight faults on edge CPUs.\[28\]
- **SENTRY (arXiv 2606.07620).** Statistical fault injection for ViT-Tiny on EuroSAT (baseline 90.99%).\[29\]\[30\]
- **RedNet.** Exploits uneven SEU/MCU sensitivity across layers.\[31\]
- **NPU-aware statistical fault injection (Electronics 15(6):1295).** Reports a 60.80% reduction in fault-injection time.\[32\]

**Checkpoint / golden-copy / correction approaches**
- **MILR (arXiv 2010.14687).** Algebraic layer recovery.\[11\]
- **CoG-guided correction (arXiv 2607.15753).** Replaces detected faulty weights with profiled min/max/mean values.\[33\]
- **Weight reconstruction (Li et al., DAC 2020; repo zlijingtao/DAC20_reconstruction).** It is bypassed by a defense-aware attacker: CFT+BR reaches 94.04% attack success (arXiv 2110.07683).\[34\] This is irrelevant for radiation, which is not adaptive.
- **RADAR (arXiv 2101.08254, DATE 2021).** 2-bit group checksums against golden signatures. It "recovers" by zeroing the flagged group, raising ResNet-18/ImageNet from <1% to >69%.\[5\]
- **US patent 12634320.** Explicitly describes replacing corrupted layers with backup copies.\[35\] Cite it in any novelty claim about "layer-selective restore".

**Self-repair / self-healing / regeneration**
- **SHNN, modular patch layers (Scientific Reports 2026, DOI 10.1038/s41598-026-57677-x).** Localizes damage by "activation discrepancy analysis at each layer" and then retrains the afflicted layer selectively, restoring accuracy to 97.33%.\[10\] This is the closest analogue to Aegis's monitor-plus-repair concept, but it needs training data.
- From your list (not re-verified by us): neural cellular automata weight repair, and rule-mining recovery of corrupted CNN bits (Roy & Ludwig 2020).

**Online / continual-learning recovery and plasticity**
- **ReDo (Sokar et al., arXiv 2302.12902, ICML 2023).** Recycles dormant neurons during RL training.\[36\]
- **Continual Backpropagation (Dohare et al., Nature 2024).** Reinitializes low-utility units.
- Both assume an ongoing training signal. Neither restores a specific learned function after weight corruption.
- **In-orbit adaptation.** OPS-SAT claimed the "first re-training of an on-board AI model with live in-flight data". SpIRIT's TFIT (CVPR 2026 AI4Space workshop) performed bounded in-orbit fine-tuning on a Jetson Nano using uplinked labels.\[37\]\[38\]

**Neuromorphic**
- **arXiv 2605.00030, "Shooting Neutrons at Neurons".** Ran the ODIN SNN on a flash-based FPGA under a neutron beam. Enabling SDSP online learning significantly extended time-to-failure and partially recovered accumulated bit flips.\[39\]
- **SpikeFI (arXiv 2412.06795).** A fault injector for SNNs.\[40\]

**BFA defenses that transfer to radiation**
- **Checksums and weight reconstruction.** RADAR; weight reconstruction; concurrent weight encoding (ICCAD 2020).\[41\]
- **Binarization / piecewise clustering.** BIN (He et al., CVPR 2020).\[41\]
- **WeightSentry (ACM 2025).** Builds on the observation that "nearly all impactful BFA vulnerabilities in FP models originate from flipping the most significant bit (MSB) of the exponent field from 0 to 1".\[42\]
- **Exponent-bit masking for LLMs (EMNLP 2025).**\[43\]
- **Transfer caveat.** These defenses transfer well because they are cheap and data-free. However, they were designed against targeted, few-bit attacks, not against random, accumulating, sometimes spatially clustered MBUs.

**Surveys**
- Lange et al., arXiv 2405.02642 (IEEE SCC 2024; repo langekevin/mlspace_robustness). Includes a negative result on NVBitFI/LLTFI usability for cloud-detection models.\[8\]\[44\]
- Ibrahim et al., "Soft errors in DNN accelerators: A comprehensive review" (Microelectronics Reliability).\[15\]
- "Resilience of deep learning applications: a systematic literature review" (Computer Science Review, 2024).\[45\]
- "AI Infrastructure in Space: How Far Can We Go?" (arXiv 2608.21034).\[12\]
- NASA TM-20220013182 on radiation tolerance and mitigation for DNNs.\[46\]

### 2. Flight heritage and beam tests

| Platform / test | What was shown | Relevance to Aegis |
|---|---|---|
| Φ-Sat-1 (Myriad 2, CloudScout) | First in-orbit DNN on an AI accelerator (2020). 92% accuracy, 1% FP. Myriad 2 functional up to 49 krad(Si). 36-hour CERN beam campaign. Per ESA's Gianluca Furano (quoted by SatNews, 27 Oct 2020), discarding cloudy images onboard saved "about 30 percent of bandwidth". | COTS accelerators survive LEO. Reliability was handled via characterization plus built-in self-test, not self-repair. |
| OPS-SAT (2019–May 2024) | 284 experiments. First in-orbit NN deployment, in-orbit training, and onboard ANN update.\[38\]\[47\] | Shows that model update/reload by uplink is operationally normal. |
| Chaohu-1 / RedNet | Jetson Xavier NX payload. Radiation errors suppressed to ≈0; 8.4–33.0% speedup. Uplink up to 1 Mbps, ~20 MB engine file.\[7\]\[31\] | Direct competitor. The uplink figures let you quantify the cost of a full model re-upload. |
| SpIRIT (Jetson Nano) | In-orbit fine-tuning with uplinked labels (TFIT).\[37\] | Label-dependent recovery is feasible but needs ground contact. |
| Jetson TX1/TX2 (NASA proton tests, 200 MeV, MGH) | Most upsets in SRAM/DRAM/bus. Recoverable by power cycle.\[48\] | Supports a "watchdog + reload" baseline. |
| Jetson Orin Nano under protons | "Characterizing Single Event Functional Interrupts in Jetson Orin Nano SoC Under Proton Irradiation" (Springer, 2026): boot/reboot is "approximately 3.3 times more susceptible to SEFIs than the inference workload", with a nominal boot of about 43 s. | Argues against frequent full resets, which supports cheaper selective repair. |
| PolarFire + Orin NX in orbit | Most observed faults came from system integration, not isolated radiation events.\[49\] | Honest caveat: SEU-in-weights may not be the dominant real failure mode. |
| DLR ScOSA design | Rad-hard co-CPU shuts down the Jetson during SAA passes.\[50\] | SAA-aware operation is already practiced at system level, but not for DNN-weight repair. |
| GPU neutron beam (Fernandes dos Santos et al.) | Real misdetection rate 8.66× higher than single-bit-flip injection.\[17\] | Your fault model must include multi-bit and logic/activation faults. |

### 3. Realistic LEO fault model: what to assume

- **Upset rates.**
  - Alsat-1 (686 km): "one error bit in one million bits per day" (~1e-6 upsets/bit/day). 80% were single-bit and 20% double- or multi-bit.\[51\]
  - Flying Laptop (~600 km, LEON3FT SRAM with EDAC): per "Analysing single event upsets in low Earth orbit considering the geomagnetic field" (Advances in Space Research, 2026; 2017–2018 data), 5,141 of 6,984 upsets (73.41%) occurred within the SAA region. The fitted rate outside the SAA is R = 1.66×10^-7 · Rc^-1.35 upsets/bit/day (Rc = geomagnetic cutoff rigidity in GV).
  - A typical SRAM background rate of ~1e-8 upsets/bit/day can rise to ~1e-6 or higher during solar particle events.\[52\]
  - Girgis et al. (Space Weather, 2023) modeled a 19% rise in SEU-rate probability during an intense geomagnetic storm (Dst = −210 nT) for a polar LEO orbit; the near-polar (i = 51°) case rose only 3%.
  - FinFET SRAMs may be more sensitive to low-energy protons (RadFxSat-2).\[53\]
- **Back-of-envelope accumulation (our estimate, not a published figure).**
  - A 1M-parameter float32 model is 32 Mbit. At 1e-6/bit/day that is ~32 flips/day unprotected; at 1e-8 it is ~0.3/day.
  - About 1/32 of flips land on the exponent MSB. Since nearly all trained weights have |w|<2, that bit is 0, so almost every such hit is catastrophic.
  - So at the high rate, expect roughly one model-breaking flip per day per million float32 parameters between scrubs.
  - ResNet-18 (~11.7M params) gives ~10×. INT8 shrinks both the bit count and the worst-case magnitude.
  - This is why scrub/repair interval, SAA timing and number format are the real experimental variables.
- **Which bits matter.**
  - float32: the exponent MSB (0→1) dominates.\[42\] The sign bit and high exponent bits come next; mantissa flips are largely benign. float16 has a narrower exponent, so the damage is bounded but still severe.
  - INT8: the damage is bounded by the quantization range. A single flip still shifts a weight by a large fraction of the dynamic range. Rakin et al. (ICCV 2019) took an 8-bit ResNet-18 on ImageNet from 69.8% to 0.1% top-1 "only through 13 bit-flips out of 93 million bits", whereas "randomly flipping 100 bits merely degrades the accuracy by less than 1%".
- **Defensible simulation assumptions.**
  - Bit-uniform random flips in stored weights at a per-bit rate drawn from the measured range above, accumulated over time.
  - A spatially clustered MBU component (adjacent bits/words, ~20% of events per the Alsat-1 data).
  - Higher rates during SAA windows.
  - Optional activation/logic faults, given the GPU beam data.
  - Explicitly not justified: "zero the weight" or "silence the neuron" as the primary SEU model. That model fits stuck-at/permanent faults and TID degradation, not SEUs.

### 4. Remote sensing specifics

- **Datasets.**
  - EuroSAT (arXiv 1709.00029): 10 Sentinel-2 classes including River, SeaLake, Forest, HerbaceousVegetation, Highway, Pasture. Shallow CNNs reach up to 89.03%.\[54\]\[55\]
  - BigEarthNet, UC Merced, and SAT-6 are the other standard land-cover sets.
  - CloudScout/Φ-Sat data and the Lange et al. cloud-detection setup are relevant for cloud screening.
- **Robustness evaluation precedents.**
  - SENTRY (ViT-Tiny/EuroSAT, mission-failure threshold τ = 50% accuracy, catastrophic τ = 20%).\[30\]
  - RedNet (3 satellite image datasets).\[56\]
  - Lange et al. (cloud detection).\[8\]
- **Gap.** We found no paper reporting class-confusion structure (e.g. River↔Highway, SeaLake↔River, Forest↔HerbaceousVegetation) under bit flips. Most report only top-1 accuracy or SDC rate. This is a cheap, genuinely novel analysis angle for an EO venue: which semantic confusions do radiation faults induce, and are they mission-critical?

### 5. Does the specific combination (a)+(b)+(c)+(d) exist?

| Work | (a) Activation/BN health monitor | (b) Detection-gated selective restore from protected/compressed copy | (c) Label-free recovery | (d) Energy/radiation-aware scheduling | Space context |
|---|---|---|---|---|---|
| BLINK (arXiv 2609.06781, 2026) | Yes (BN stats) | No (re-centers activations; no reload) | Yes | No | No |
| RADAR (DATE 2021) | No (weight checksums) | Partial (zeroes flagged group) | Yes | No | No |
| Liu et al. checksums (MR 2022) | Partial (output checksums) | Detection only | n/a | No | No |
| MILR (2020) | No | Layer recovery via algebra, not copy | Yes (needs stored I/O) | No | No |
| CoG correction (2026) | No (weight ranges) | In-place substitute, not restore | Yes | No | No |
| SHNN (Sci Rep 2026) | Yes (activation discrepancy) | Retrains flagged layer | No (needs data) | No | No |
| Ranger / FT-ClipAct / ProAct | Implicit (bounds) | No | Yes | No | No |
| RedNet (2024) | No | No (clipping + early exit) | Yes | No | Yes (Chaohu-1) |
| ZEDAC (SmallSat 2026) | No (EDAC metadata) | Correction on read, not detection-gated by behavior | Yes | No | Yes |
| BN recalibration (arXiv 2010.05625; FPT patent) | No | No | Yes (unlabeled; ~1,024 samples in FPT)\[57\]\[58\] | No | No |
| US patent 12634320 | Hash/parity checks | Yes (backup layer replacement) | Yes | No | No |
| Adaptive FPGA scrubbing (DAC 2015) | No | Config scrubbing | n/a | Partial (adaptive rate) | Yes |
| **Proposed Aegis-v2** | Yes | Yes | Yes | Yes | Yes |

**Verdict.** Every individual piece exists. The combination, and especially (d) radiation/energy-aware scheduling of DNN repair, appears unoccupied in peer-reviewed work. Confidence is moderate: we did not exhaustively search IEEE TNS, RADECS or NSREC proceedings, and BLINK-style work is moving fast (it is only six weeks old).

Two critiques also need answering:
- **"Why not just scrub everything?"** The paper must show that detection-gated selective reload beats periodic full scrubbing on energy/latency or on accuracy-over-time at equal budget.
- **"Why not ECC?"** ECC protects memory but not logic or activations (beam data: ECC was ineffective in reducing critical errors).\[18\]\[59\] In practice ECC may also be unavailable on COTS SD/eMMC or in plaintext paths.

### 6. Open-source tools and repositories

| Tool / repo | What it does | Overlap with Aegis |
|---|---|---|
| PyTorchFI | Weight/neuron perturbation injection in PyTorch\[60\] | Replace your custom injector with this; reviewers trust it. |
| PyTorchALFI (IntelLabs) | Application-level FI built on PyTorchFI, with KPIs for classification/detection | Good for campaign management and metrics. |
| Ares (Reagen et al., DAC 2018) | DNN FI, validated within 12% of a real accelerator\[61\] | Use as a citation for application-level FI validity. |
| TensorFI / TensorFI2 / TensorFI+ / BinFI (DependableSystemsLab) | TensorFlow FI; BinFI finds safety-critical bits\[60\] | BinFI's bit criticality supports the exponent-MSB argument. |
| NVBitFI (NVIDIA; DSN 2021) and radhelper fork with FP16/Tensor Core/warp fault models | GPU instruction-level FI\[62\] | Needed only if you target Jetson realism. |
| MRFI (arXiv 2306.11758) | Multi-resolution FI\[63\] | Alternative to PyTorchFI. |
| SpikeFI | SNN FI\[40\] | Only if you pivot to neuromorphic. |
| langekevin/mlspace_robustness | Cloud-detection bit-flip experiments (SCC 2024)\[64\] | Closest space-ML baseline code. |
| zlijingtao/DAC20_reconstruction | Weight reconstruction defense\[65\] | Cheap baseline. |
| RedNet radiation-error emulator (anonymous.4open.science) | Real-data-driven radiation emulator\[56\] | Check whether it is still accessible; ideal for the fault model. |
| r0nlt/Space-Radiation-Tolerant, ss1738/euroSAT-CNN-landcover | Hobby/unreviewed projects | Do not cite as evidence; at most mention as "community efforts". |

## Candid assessment of current Aegis weaknesses

1. **Fault-model mismatch.** Zeroing weights or suppressing outputs models stuck-at faults, not SEUs. Real float32 SEUs create huge activations (inf/NaN risk), which a "below-ε firing" monitor will never flag. In fact, a dead-neuron monitor watches for the opposite symptom.
2. **The repair tier cannot restore function.** Noise-shock revival and micro-resets (ReDo/Continual Backprop) re-randomize capacity; they do not recover the specific learned mapping without gradient signal. Homeostatic rescaling is similar in spirit to BN recalibration, which helps only with distribution shift, not arbitrary weight corruption. With a golden copy available, reload dominates.
3. **Missing baselines.** No ECC/scrubbing, full reload, Ranger/clipping, BLINK, MILR or selective TMR comparison.
4. **MLP-only.** Results on an MLP do not transfer to CNNs or ViTs (BN, residual paths, depth). Reviewers will discount them.
5. **No time dimension.** Faults accumulate between scrubs; a single-shot injection ignores the core operational question.
6. **The dashboard stack is not a contribution.** Streamlit/FastAPI/SQLite telemetry is fine for a demo but should not appear in the contributions list.

## Recommendations: ranked reframings

Ranking for undergraduates over a few months: A (most feasible) > E > B > C > D. Ranking by novelty: B > C > D > A > E.

**A. "Do activation-health monitors detect realistic radiation faults? A fault-model-grounded study for onboard EO classifiers"**
- *Problem:* Onboard health monitors are evaluated under unrealistic fault models.
- *RQ:* Under LEO-calibrated accumulated bit flips (float32/float16/INT8, ~20% MBU, SAA-modulated), what detection latency, precision and recall do dead-neuron monitors, range monitors (Ranger bounds), BN-statistic monitors (BLINK) and weight checksums achieve? Which class confusions emerge?
- *Baselines:* the four detectors above, plus no-protection.
- *Setup:* ResNet-18 and MobileNetV2 (optionally ViT-Tiny) on EuroSAT, with UC Merced as a second dataset. Use PyTorchFI with a custom time-stepped accumulator. Metrics: accuracy over simulated days, detection AUROC, time-to-detect, confusion-matrix deltas.
- *Contribution:* a negative or positive result plus an open benchmark. It directly reuses Aegis's monitor.
- *Venues:* DSN workshops (e.g. DSML), IEEE SCC/SMC-IT, CVPR AI4Space workshop, arXiv-first.
- *Risk:* low novelty if framed as "just a benchmark". Mitigate with the class-confusion and fault-model-sensitivity findings.

**B. "Detection-gated, layer-selective restore under an energy budget"** (the main novelty)
- *Problem:* Full periodic scrubbing or re-upload wastes energy and uplink; pure in-situ mitigation (clipping/BLINK) leaves corrupted weights in place.
- *RQ:* Can a cheap monitor trigger reload of only flagged layers from a compressed (e.g. INT8 + ECC or zstd) protected copy, achieving equal or better accuracy-over-time at lower energy/bytes than periodic full scrubbing?
- *Baselines:* periodic full scrub at several intervals, ECC-only, Ranger, BLINK, MILR (if feasible), RADAR-style group checksums, and full model re-upload (costed with Chaohu-1's ~1 Mbps / ~20 MB figures).
- *Metrics:* area under the accuracy-vs-time curve; bytes read/written; energy proxy (memory traffic × per-byte energy constants); unavailability windows.
- *Contribution:* a new point on the cost/reliability Pareto front.
- *Venues:* SmallSat, IEEE Aerospace, DATE/DAC workshops, IOLTS.
- *Risk:* the energy model is a proxy. Any Raspberry Pi or Jetson power measurement would greatly strengthen it.

**C. "Orbit-aware repair scheduling"**
- *Problem:* Fault arrival is non-uniform; per Advances in Space Research (2026), 73.41% (5,141 of 6,984) of Flying Laptop upsets occurred in the SAA.
- *RQ:* Does scheduling scrub/reload around SAA exits and solar-event alerts beat fixed-interval scrubbing at equal budget?
- *Setup:* As in B, with a fault-arrival process driven by orbit position (use Flying Laptop/Alsat-1 rates, or AP8/CREME-style rate tables if accessible).
- *Contribution:* the first DNN-specific orbit-aware repair policy.
- *Risk:* rests on simulation only; reviewers from the radiation community may question the rate model.

**D. "Label-free recovery when the golden copy is unavailable or also corrupted"**
- *RQ:* For permanent or stuck-at degradation (TID, aging), where reload does not help, how much accuracy do BN recalibration on unlabeled onboard imagery, BLINK-style re-centering, and self-distillation from a smaller protected model recover?
- *Contribution:* this is where Aegis's "homeostatic" idea is legitimately useful, because stuck-at faults are the one case where a zero/stuck fault model is correct.
- *Risk:* overlaps with existing in-memory-computing BN-calibration work.\[66\] Must be framed explicitly as the space/permanent-fault case.

**E. "Does plasticity-style neuron revival help after hardware faults?"**
- *RQ:* Compare ReDo/continual-backprop resets against reload and clipping under realistic faults.
- *Contribution:* a likely negative result that cleanly salvages existing Aegis code.
- *Venues:* workshop or arXiv.
- *Risk:* low novelty, though easy to execute.

**Recommended path.** Do A as the paper core and include a B prototype as the "proposed method" section. That one paper, "Benchmark + detection-gated selective restore", is realistic for undergraduates and has a clear contribution.

## Minimal 3-week plan for the final review

- **Week 1 (fault model + baselines).**
  - Swap the custom injector for PyTorchFI.
  - Implement a time-stepped accumulator with per-bit rates 1e-8 to 1e-6/bit/day, ~20% adjacent-bit MBUs, and an SAA multiplier.
  - Train ResNet-18 on EuroSAT.
  - Reproduce a single figure: accuracy vs simulated days for float32 vs INT8, with no protection.
  - Show the exponent-MSB effect (histogram of post-fault activation magnitudes vs dead-neuron counts). This one plot answers the fault-model critique.
- **Week 2 (detectors + simple restore).**
  - Implement four detectors: dead-neuron (existing), activation-range (Ranger bounds), BN-statistic drift (BLINK-style), and per-layer CRC.
  - Implement two repairs: full periodic reload, and detection-gated per-layer reload from an INT8 copy.
  - Add the Ranger clipping baseline.
- **Week 3 (write-up).**
  - Build a related-work table (use the table in Section 5).
  - Write a one-paragraph problem statement: "Onboard EO classifiers on COTS hardware accumulate radiation-induced weight corruption between scrub/uplink cycles; we study whether behavior-based detection can gate selective, low-cost restoration, and evaluate under LEO-calibrated fault models."
  - Produce two result figures (accuracy-over-time; detection AUROC per detector) and the confusion-matrix delta for River/SeaLake/Highway.
  - Write a limitations slide.

## Caveats

- Several 2026 items (BLINK, CoG, SPINE, SENTRY, SHNN, ZEDAC, the SpIRIT TFIT paper, "Shooting Neutrons at Neurons") are very recent preprints or proceedings. Their numbers are as stated in the abstracts and have not been independently replicated.
- Items from your list that we did not re-verify are labeled as such: TR-ABFT, N-Slim, neural cellular automata repair, Roy & Ludwig, Global Clipper details, and arXiv 2411.18993.
- The upset-rate figures come from specific SRAMs and orbits. Jetson-class LPDDR with ECC behaves differently. The "one catastrophic flip per day per million parameters" figure is our arithmetic, not a measured result.
- Real in-orbit evidence (the PolarFire + Orin NX validation) suggests system-integration faults and SEFIs may dominate over SEUs in weights.\[49\] Acknowledge this as a threat to validity.
- The novelty verdict on (a)+(b)+(c)+(d) reflects our search scope. Re-run a targeted search (IEEE TNS, RADECS, NSREC, IOLTS, and arXiv "selective reload" / "golden model" / "scrubbing DNN") within a week of submission, since this area is moving monthly.

## Sources

1. [(PDF) Program memories error detection and correction on-board earth observation satellites](https://www.researchgate.net/publication/289589794_Program_memories_error_detection_and_correction_on-board_earth_observation_satellites)
2. [NVIDIA Jetson Orin on Satellites: What On-Orbit Edge AI Looks Like Today](https://hubble.com/community/guides/nvidia-jetson-orin-on-satellites-what-on-orbit-edge-ai-looks-like-today/)
3. [A Low-cost Fault Corrector for Deep Neural Networks through Range Restriction](https://arxiv.org/abs/2003.13874)
4. [FT-ClipAct: Resilience Analysis of Deep Neural Networks and Improving their Fault Tolerance using Clipped Activation](https://www.catalyzex.com/paper/ft-clipact-resilience-analysis-of-deep-neural)
5. mcp\_\_Consensus\_\_search
6. [BLINK: Batch Normalization-based Integrity Checkpoints for In-Situ Detection and Mitigation of Diverse Weight Corruptions in DNN Accelerators](https://arxiv.org/abs/2609.06781)
7. [A Case for Application-Aware Space Radiation Tolerance in Orbital Computing](https://arxiv.org/pdf/2407.11853)
8. [Machine Learning in Space: Surveying the Robustness of on-board ML models to Radiation](https://arxiv.org/pdf/2405.02642)
9. [DigitalCommons@USU - Small Satellite Conference: Radiation-Resilient AI for Small Satellites: A Universal Multi-Bit EDAC Architecture for Protecting On-Board Neural Network Models in Memory and Storage](https://digitalcommons.usu.edu/smallsat/2026/all2026/51/)
10. [Self-healing neural networks via modular patch layers for diverse structural and adversarial damages - PubMed](https://pubmed.ncbi.nlm.nih.gov/42331864/)
11. <https://arxiv.org/pdf/2010.14687>
12. [AI Infrastructure in Space: How Far Can We Go?](https://arxiv.org/pdf/2608.21034)
13. [www.ece.nus.edu.sg](https://www.ece.nus.edu.sg/stfpage/eleak/pdf/dac15.pdf)
14. [An ECC-based Fault Tolerance Approach for DNNs](https://arxiv.org/pdf/2508.12347)
15. [In-Situ Hardware Error Detection Using Specification-Derived Petri Net Models and Behavior-Derived State Sequences](https://arxiv.org/pdf/2505.04108)
16. [Safe-NEureka: a Hybrid Modular Redundant DNN Accelerator for On-board Satellite AI Processing](https://arxiv.org/abs/2602.04803)
17. [\[2211.13094\] Characterizing a Neutron-Induced Fault Model for Deep Neural Networks](https://arxiv.org/abs/2211.13094)
18. [Characterizing a Neutron-Induced Fault Model for Deep Neural Networks - ADS](https://ui.adsabs.harvard.edu/abs/2023ITNS...70..370S/abstract)
19. [Global Clipper: Enhancing Safety and Reliability of Transformer-based Object Detection Models](https://arxiv.org/pdf/2406.03229)
20. [ProAct: Progressive Training for Hybrid Clipped Activation ...](https://arxiv.org/pdf/2406.06313)
21. [Software-Implemented Fault Tolerance for Model Inference ...](https://www.usenix.org/system/files/atc25-zheng.pdf)
22. [Enhancing Neural Network Robustness Against Fault Injection Through Non-linear Weight Transformations · Pith](https://pith.science/paper/2411.19027)
23. [FAQ: Mitigating the Impact of Faults in the Weight Memory of DNN Accelerators through Fault-Aware Quantization](https://arxiv.org/pdf/2305.12590)
24. [(PDF) FAT-RABBIT: Fault-Aware Training towards Robustness Against Bit-flip Based Attacks in Deep Neural Networks](https://www.researchgate.net/publication/385101469_FAT-RABBIT_Fault-Aware_Training_towards_Robustness_Against_Bit-flip_Based_Attacks_in_Deep_Neural_Networks)
25. [VulnerabIlity Value RanGes and FactORs for DNNs' ...](https://arxiv.org/pdf/2303.06931)
26. [DeepVigor: VulnerabIlity Value RanGes and FactORs for DNNs’ Reliability Assessment](https://www.researchgate.net/publication/372324599_DeepVigor_VulnerabIlity_Value_RanGes_and_FactORs_for_DNNs'_Reliability_Assessment)
27. [(PDF) DeepVigor+: Scalable and Accurate Semi-Analytical Fault Resilience Analysis for Deep Neural Network](https://www.researchgate.net/publication/385107677_DeepVigor_Scalable_and_Accurate_Semi-Analytical_Fault_Resilience_Analysis_for_Deep_Neural_Network)
28. [SPINE: A Fault Injection Profiler for Quantized Neural Networks under Accumulated Faults](https://arxiv.org/abs/2606.19526)
29. [Statistical Reliability Analysis of Vision Transformers Under ...](https://arxiv.org/pdf/2606.07620)
30. [SENTRY: Statistical Reliability Analysis of Vision Transformers Under Soft Errors](https://arxiv.org/html/2606.07620)
31. [\[2407.11853\] A Case for Application-Aware Space Radiation Tolerance in Orbital Computing](https://arxiv.org/abs/2407.11853)
32. [NPU-Aware Fault Injection and Statistical Sensitivity Analysis for CNN Reliability Evaluation](https://doi.org/10.3390/electronics15061295)
33. [CoG-Guided Weight Correction for Fault-Tolerant Deep Neural Networks · Pith Review](https://pith.science/paper/2607.15753)
34. [Don't Knock! Rowhammer at the Backdoor of DNN Models](https://arxiv.org/pdf/2110.07683)
35. [System and method for providing advanced neuron-level vulnerability analysis in neural network-based models](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12634320)
36. [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/abs/2302.12902)
37. [Flight Demonstration of On-Orbit Model Adaptation on the SpIRIT Nanosatellite](https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/papers/Joukhadar_Flight_Demonstration_of_On-Orbit_Model_Adaptation_on_the_SpIRIT__CVPRW_2026_paper.pdf)
38. [OPS-SAT - ESOC - European Space Agency](https://esoc.esa.int/content/ops-sat)
39. [Shooting Neutrons at Neurons: Radiation Testing of a Spiking Neural Network on Flash-Based FPGAs](https://arxiv.org/abs/2605.00030)
40. [SpikeFI: A Fault Injection Framework for Spiking Neural Networks](https://arxiv.org/pdf/2412.06795)
41. [Aegis: Mitigating Targeted Bit-flip Attacks against Deep Neural Networks](https://www.usenix.org/system/files/sec23fall-prepub-246-wang-jialai.pdf)
42. [WeightSentry: Real-Time Bit-Flip Protection for Deep Neural Networks on GPUs](https://dl.acm.org/doi/10.1145/3768725.3768734)
43. [Bit-Flip Error Resilience in LLMs](https://aclanthology.org/2025.emnlp-main.528.pdf)
44. [\[Literature Review\] Machine Learning in Space: Surveying the Robustness of on-board ML models to Radiation](https://www.themoonlight.io/en/review/machine-learning-in-space-surveying-the-robustness-of-on-board-ml-models-to-radiation)
45. [Resilience of deep learning applications: A systematic literature review of analysis and hardening techniques - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1574013724000662)
46. [NASA/TM–20220013182 Radiation Tolerance and Mitigation for](https://ntrs.nasa.gov/api/citations/20220013182/downloads/Rad-tolerance-report-2022.pdf)
47. [OPS-SAT (Operations nanoSatellite) - eoPortal](https://www.eoportal.org/satellite-missions/ops-sat)
48. [To be published on nepp.nasa.gov. Proton Testing of nVidia Jetson TX1](https://ntrs.nasa.gov/api/citations/20170009004/downloads/20170009004.pdf)
49. [In-Orbit Validation of a Heterogeneous COTS Architecture for Onboard AI: System-Level Operational Vulnerabilities and Failure Pathways](https://link.springer.com/article/10.1007/s42405-026-01130-w)
50. [Feasibility of a COTS GPU-Based On-board AI Platform for ...](https://elib.dlr.de/218458/1/2025_IAC_spAIce.release.pdf)
51. [(PDF) Observations of single-event upsets and multiple-bit upsets in random access memories on-board the Algerian satellite](https://www.researchgate.net/publication/224380922_Observations_of_single-event_upsets_and_multiple-bit_upsets_in_random_access_memories_on-board_the_Algerian_satellite)
52. [Electronic circuit with integrated SEU monitor](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11586496)
53. [The RadFxSat-2 Mission to Measure SEU Rates in FinFET ...](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=5576&context=smallsat)
54. [GitHub - MSEldeeb/eurosat-cnn-classifier: This project explores image classification using CNNs with TensorFlow/Keras on the EuroSAT Sentinel-2 satellite imagery dataset. It covers data exploration, CNN development, data augmentation, model evaluation, and error analysis, achieving 96.67% test accuracy across 10 land-use and land-cover classes. · GitHub](https://github.com/MSEldeeb/eurosat-cnn-classifier)
55. [EuroSAT: A Novel Dataset and Deep Learning Benchmark ...](https://arxiv.org/pdf/1709.00029)
56. [A Case for Application-Aware Space Radiation Tolerance in Orbital Computing](https://arxiv.org/html/2407.11853v1)
57. [arXiv:2010.05625v1 \[cs.LG\] 12 Oct 2020 Post-Training BatchNorm Recalibration](https://arxiv.org/pdf/2010.05625)
58. [Stuck-at fault mitigation method for ReRAM-based deep learning accelerators](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11714727)
59. [Evaluation and Mitigation of Soft-Errors in Neural Network-Based Object Detection in Three GPU Architectures](https://www.researchgate.net/publication/319409068_Evaluation_and_Mitigation_of_Soft-Errors_in_Neural_Network-Based_Object_Detection_in_Three_GPU_Architectures)
60. [Resilience of Deep Learning applications: a systematic survey of analysis and](https://scispace.com/pdf/resilience-of-deep-learning-applications-a-systematic-survey-w0mzj1f01q.pdf)
61. [Ares: A framework for quantifying the resilience of deep neural networks](https://dl.acm.org/doi/pdf/10.1145/3195970.3195997)
62. [Characterizing a Neutron-Induced Fault Model for Deep Neural Networks](https://arxiv.org/pdf/2211.13094)
63. [MRFI: An Open Source Multi-Resolution Fault Injection ...](https://arxiv.org/pdf/2306.11758)
64. [GitHub - langekevin/mlspace\_robustness: Artifacts of the paper "Machine Learning in Space: Surveying the Robustness of on-board ML models to Radiation" accepted to the IEEE Space Computing Conference 2024 · GitHub](https://github.com/langekevin/mlspace_robustness/)
65. [Defending Bif-Flip Attack using Weight Reconstruction](https://github.com/zlijingtao/DAC20_reconstruction)
66. [Examining the Role and Limits of Batchnorm Optimization to Mitigate Diverse Hardware-noise in In-memory Computing](https://www.researchgate.net/publication/371305984_Examining_the_Role_and_Limits_of_Batchnorm_Optimization_to_Mitigate_Diverse_Hardware-noise_in_In-memory_Computing)
