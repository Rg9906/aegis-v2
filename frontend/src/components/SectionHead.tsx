import type { ReactNode } from "react";
import { motion } from "motion/react";

interface Props {
  index: string;
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
}

export default function SectionHead({ index, kicker, title, lede }: Props) {
  return (
    <motion.header
      className="section-head"
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.7, ease: [0.2, 0.7, 0.1, 1] }}
    >
      <div className="kicker">
        <span className="kicker-index mono">{index}</span>
        <span>{kicker}</span>
      </div>
      <h2>{title}</h2>
      {lede && <p className="section-lede">{lede}</p>}
    </motion.header>
  );
}
