import type { StrategyKey } from "./types";

// Checkpoint logit order (aegis/config.py EUROSAT_CLASSES), NOT alphabetical
export const CLASSES = [
  "Forest",
  "River",
  "Highway",
  "AnnualCrop",
  "SeaLake",
  "HerbaceousVegetation",
  "Industrial",
  "Residential",
  "PermanentCrop",
  "Pasture",
] as const;

export const CLASS_SHORT = ["Forest", "River", "Highway", "Crop", "Sea/Lake", "Herbaceous", "Industrial", "Residential", "Perm. crop", "Pasture"];

export const STRATEGY_COLOR: Record<StrategyKey, string> = {
  unmitigated: "var(--s-acc_unmitigated)",
  scrub: "var(--s-acc_periodic_scrub)",
  aegis: "var(--s-acc_aegis)",
  audit: "var(--s-acc_aegis_audit)",
};

export const FIELD_INFO = {
  exponent_msb: {
    label: "Exponent MSB",
    tag: "bit 30",
    blurb: "Top exponent bit: scales the weight by 2¹²⁸. The flip that breaks networks.",
  },
  exponent: { label: "Exponent", tag: "bits 23–30", blurb: "Any exponent bit: scales the value by a power of two." },
  mantissa: { label: "Mantissa", tag: "bits 0–22", blurb: "Fraction bits: tiny nudges, usually invisible to the model." },
  sign: { label: "Sign", tag: "bit 31", blurb: "Negates the value. Fatal on a BatchNorm variance (√ of a negative)." },
  any: { label: "Any bit", tag: "bits 0–31", blurb: "Uniform over all 32 bits, like a real particle strike." },
} as const;
