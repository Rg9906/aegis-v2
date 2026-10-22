/**
 * Looping schematic of one Aegis-v2 repair cycle: imagery flows through the model, an upset
 * strikes L3, T1 raises an alarm, T3's CRC confirms the mismatch, and the golden copy restores
 * only that block. Timing lives in CSS (.pipe-* keyframes, one 9 s cycle).
 */
const BLOCKS = ["Stem", "L1", "L2", "L3", "L4", "FC"];
const X0 = 150;
const STEP = 104;
const W = 80;
const HIT = 3;

const CAPTIONS = [
  ["cap-0", "Nominal: hooks watch every layer on each image batch"],
  ["cap-1", "A heavy ion flips an exponent bit in L3"],
  ["cap-2", "T1: L3 activations leave their calibrated envelope"],
  ["cap-3", "T3: CRC-32 of L3 no longer matches the golden checksum"],
  ["cap-4", "Restore L3 only, bit-exact, from the zstd golden copy"],
  ["cap-5", "Healed. The other five blocks never stopped"],
];

export default function Pipeline() {
  const hx = X0 + HIT * STEP;
  return (
    <div className="panel pipeline" role="img" aria-label="Animated schematic of an Aegis-v2 detect, confirm and restore cycle">
      <svg viewBox="0 0 1000 300" className="pipe-svg">
        <defs>
          <linearGradient id="pipe-flow" x1="0" x2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity="0" />
            <stop offset="0.5" stopColor="var(--accent)" stopOpacity="0.55" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
          <filter id="pipe-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        {/* Image batch */}
        <g className="pipe-input">
          {[0, 1, 2].map((i) => (
            <g key={i} transform={`translate(${28 + i * 7} ${102 + i * 7})`}>
              <rect width="58" height="58" rx="6" className="pipe-chip" />
              <rect x="6" y="6" width="22" height="22" rx="2" className="pipe-chip-a" />
              <rect x="30" y="30" width="22" height="22" rx="2" className="pipe-chip-b" />
            </g>
          ))}
          <text x="58" y="200" className="pipe-label" textAnchor="middle">EuroSAT batch</text>
        </g>

        {/* Data bus with moving packets */}
        <line x1="108" y1="140" x2="790" y2="140" className="pipe-bus" />
        <rect x="108" y="138" width="682" height="4" fill="url(#pipe-flow)" className="pipe-sheen" />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} r="3.5" className="pipe-packet">
            <animateMotion dur="2.4s" begin={`${i * 0.6}s`} repeatCount="indefinite" path="M108 140 H790" />
          </circle>
        ))}

        {/* Model blocks + T1/T2 sensors */}
        {BLOCKS.map((b, i) => {
          const x = X0 + i * STEP;
          const hit = i === HIT;
          return (
            <g key={b}>
              <line x1={x + W / 2} y1="78" x2={x + W / 2} y2="112" className={`pipe-probe ${hit ? "pipe-probe-hit" : ""}`} />
              <circle cx={x + W / 2} cy="70" r="7" className={`pipe-sensor ${hit ? "pipe-sensor-hit" : ""}`} />
              <rect x={x} y="112" width={W} height="56" rx="10" className={`pipe-block ${hit ? "pipe-block-hit" : ""}`} />
              <text x={x + W / 2} y="146" textAnchor="middle" className="pipe-block-label">{b}</text>
            </g>
          );
        })}
        <text x={X0 - 14} y="74" textAnchor="end" className="pipe-label">T1 / T2</text>

        {/* Strike */}
        <g className="pipe-bolt">
          <path d={`M${hx + 52} 0 L${hx + 30} 52 L${hx + 46} 52 L${hx + 26} 108`} className="pipe-bolt-path" filter="url(#pipe-glow)" />
          <path d={`M${hx + 52} 0 L${hx + 30} 52 L${hx + 46} 52 L${hx + 26} 108`} className="pipe-bolt-path" />
        </g>

        {/* T3 CRC badge */}
        <g className="pipe-crc-bad">
          <rect x={hx + 4} y="178" width="72" height="22" rx="11" className="pipe-badge-bad" />
          <text x={hx + 40} y="193" textAnchor="middle" className="pipe-badge-text">CRC ✗</text>
        </g>
        <g className="pipe-crc-good">
          <rect x={hx + 4} y="178" width="72" height="22" rx="11" className="pipe-badge-good" />
          <text x={hx + 40} y="193" textAnchor="middle" className="pipe-badge-text">CRC ✓</text>
        </g>

        {/* Golden copy in flash, restore beam */}
        <g transform="translate(820 214)">
          <ellipse cx="60" cy="0" rx="58" ry="11" className="pipe-flash-top" />
          <path d="M2 0 V44 A58 11 0 0 0 118 44 V0" className="pipe-flash" />
          <text x="60" y="30" textAnchor="middle" className="pipe-flash-text">golden copy</text>
          <text x="60" y="76" textAnchor="middle" className="pipe-label">flash · zstd · CRC table</text>
        </g>
        <path d={`M820 226 C ${hx + 220} 270, ${hx + 60} 260, ${hx + 40} 202`} className="pipe-beam" pathLength={100} />

        {/* Output */}
        <g transform="translate(808 112)">
          <rect width="150" height="56" rx="10" className="pipe-out" />
          <text x="75" y="25" textAnchor="middle" className="pipe-label">prediction</text>
          <text x="75" y="44" textAnchor="middle" className="pipe-out-text">Forest · 0.97</text>
        </g>
      </svg>
      <div className="pipe-captions mono" aria-hidden="true">
        {CAPTIONS.map(([cls, text], i) => (
          <span key={cls} className={`pipe-cap ${cls}`}>
            <b>0{i + 1}</b> {text}
          </span>
        ))}
      </div>
    </div>
  );
}
