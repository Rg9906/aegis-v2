import type { Campaign } from "../lib/types";
import { sci } from "../lib/format";
import SectionHead from "./SectionHead";

type Basis = "measured" | "calibrated" | "modelled" | "assumed" | "caveat";

const BASIS_LABEL: Record<Basis, string> = {
  measured: "Real data",
  calibrated: "Calibrated to flight data",
  modelled: "Physical model",
  assumed: "Assumption",
  caveat: "Caveat",
};

export default function Method({ campaign }: { campaign: Campaign }) {
  const { setup } = campaign;
  const ratio = setup.rates.saa_core / setup.rates.background;
  const items: { basis: Basis; title: string; body: string }[] = [
    {
      basis: "measured",
      title: "Orbit",
      body: campaign.track
        ? `Sentinel-2A two-line elements from CelesTrak, propagated with SGP4 every minute (${campaign.track.alt_km.toFixed(0)} km, ${campaign.track.period_min.toFixed(1)} min period).`
        : "Synthetic periodic orbit with fixed SAA passes.",
    },
    {
      basis: "measured",
      title: "Imagery",
      body: `Real EuroSAT RGB chips (Sentinel-2), 224 px bicubic. Evaluation chips are disjoint from the chips used to calibrate the monitors. Clean accuracy ${(setup.clean_accuracy * 100).toFixed(1)}%.`,
    },
    {
      basis: "calibrated",
      title: "Upset rate",
      body: `Orbit-average ${sci(setup.rates.measured_mean)} upsets per bit per day, ${(setup.rates.sbu_fraction * 100).toFixed(2)}% single-bit: Alsat-1's 8-year SRAM record (Bentoutou 2011). Faults land uniformly over all stored bits.`,
    },
    {
      basis: "modelled",
      title: "South Atlantic Anomaly",
      body: "Where the satellite is matters: the rate ramps from background to SAA-core as the IGRF field strength at the satellite drops from 20,000 to 18,000 nT, integrated minute by minute.",
    },
    {
      basis: "assumed",
      title: "SAA intensity",
      body: `The SAA-core rate is ${ratio.toFixed(0)}× the background rate. This ratio is not measured; transfer from Alsat-1 (686 km, 2002 SRAM) to Sentinel-2A (786 km, modern memory) is also assumed.`,
    },
    {
      basis: "assumed",
      title: "Energy",
      body: `DRAM reads ${setup.energy.dram_read_pj_per_bit} pJ/bit; monitor activation checks ${setup.energy.monitor_pj_per_op} pJ/op (${setup.energy.inference_mj_per_image} mJ per inference ÷ 1.82 G MACs). For small models the monitoring cost can rival frequent scrubbing, so this constant matters.`,
    },
    {
      basis: "caveat",
      title: "Checkpoint",
      body: setup.arch === "mobilenet_v2"
        ? "MobileNetV2 fine-tuned in-project on chips held out from every campaign split; the campaign refuses to run if they overlap."
        : "The public ResNet-18 checkpoint does not publish its training split, so clean accuracy may be optimistic. Accuracy lost to radiation is unaffected.",
    },
    {
      basis: "caveat",
      title: "Not modelled",
      body: "AP8/AE9 trapped-particle spectra and device cross-sections; Ranger and BLINK per-inference compute; faults in activations are studied separately (0% spurious restores in 40 trials).",
    },
  ];

  return (
    <section className="section" id="method">
      <SectionHead
        index="06"
        kicker="Method & assumptions"
        title={<>What is <em>measured</em>, and what is assumed.</>}
        lede="Every number on this page comes from a seeded, reproducible campaign. Two independent GPU runs of the first campaign gave bit-identical results for every seed."
      />
      <div className="method-grid">
        {items.map((it) => (
          <div className="panel method-card" key={it.title}>
            <span className={`basis basis-${it.basis}`}>{BASIS_LABEL[it.basis]}</span>
            <h3>{it.title}</h3>
            <p>{it.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
