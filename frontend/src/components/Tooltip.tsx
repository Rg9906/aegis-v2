import type { ReactNode } from "react";

export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** Floating tooltip positioned in the coordinate space of its relatively positioned parent. */
export default function Tooltip({ tip, width }: { tip: TipState | null; width: number }) {
  if (!tip) return null;
  const flip = tip.x > width - 260;
  return (
    <div
      className="tip"
      style={{
        left: flip ? undefined : tip.x + 14,
        right: flip ? width - tip.x + 14 : undefined,
        top: tip.y + 14,
      }}
      role="status"
    >
      {tip.content}
    </div>
  );
}
