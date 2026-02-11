import { useMemo } from "react";

interface CoreProps {
  state: "idle" | "listening" | "thinking" | "streaming" | "error";
  healthRatio: number;
}

/**
 * The AI Entity — a luminous plasma orb that embodies the AI's presence.
 * Breathing when idle, rippling when thinking, flowing when streaming.
 */
export default function Core({ state, healthRatio }: CoreProps) {
  const colors = useMemo(() => {
    if (state === "error") return { primary: "#ef4444", secondary: "#dc2626", glow: "rgba(239,68,68,0.3)" };
    const h = healthRatio;
    // Interpolate between amber (degraded) and cyan (healthy)
    if (h > 0.7) return { primary: "#00d4ff", secondary: "#8b5cf6", glow: "rgba(0,212,255,0.25)" };
    if (h > 0.4) return { primary: "#eab308", secondary: "#00d4ff", glow: "rgba(234,179,8,0.2)" };
    return { primary: "#f97316", secondary: "#ef4444", glow: "rgba(249,115,22,0.25)" };
  }, [state, healthRatio]);

  const stateClass = {
    idle: "animate-[breathe_4s_ease-in-out_infinite]",
    listening: "animate-[breathe_2.5s_ease-in-out_infinite] scale-105",
    thinking: "animate-[ripple_1.2s_ease-in-out_infinite]",
    streaming: "animate-[pulse-stream_0.8s_ease-in-out_infinite] scale-110",
    error: "animate-[breathe_3s_ease-in-out_infinite] opacity-60",
  }[state];

  return (
    <div className="relative flex items-center justify-center" style={{ zIndex: 1 }}>
      {/* Outer glow rings */}
      <div
        className={`absolute rounded-full blur-3xl transition-all duration-1000 ${state === "streaming" ? "w-80 h-80" : "w-56 h-56"}`}
        style={{ background: `radial-gradient(circle, ${colors.glow}, transparent 70%)` }}
      />
      <div
        className={`absolute rounded-full blur-xl transition-all duration-700 ${state === "streaming" ? "w-52 h-52" : "w-36 h-36"}`}
        style={{ background: `radial-gradient(circle, ${colors.glow}, transparent 60%)` }}
      />

      {/* Orbital rings */}
      <div
        className={`absolute w-48 h-48 rounded-full border transition-all duration-500 ${state === "streaming" ? "animate-[orbit_3s_linear_infinite] opacity-40" : state === "thinking" ? "animate-[orbit_5s_linear_infinite] opacity-30" : "opacity-10"}`}
        style={{ borderColor: colors.primary, borderWidth: "1px" }}
      />
      <div
        className={`absolute w-64 h-64 rounded-full border transition-all duration-500 ${state === "streaming" ? "animate-[orbit-reverse_4s_linear_infinite] opacity-30" : "opacity-5"}`}
        style={{ borderColor: colors.secondary, borderWidth: "0.5px" }}
      />

      {/* Core sphere */}
      <div className={`relative w-28 h-28 rounded-full transition-all duration-500 ${stateClass}`}>
        {/* Primary gradient sphere */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: `radial-gradient(circle at 35% 35%, ${colors.primary}cc, ${colors.secondary}88, transparent)`,
            boxShadow: `0 0 60px ${colors.glow}, inset 0 0 40px ${colors.glow}`,
          }}
        />
        {/* Glass highlight */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: "radial-gradient(circle at 30% 25%, rgba(255,255,255,0.25), transparent 50%)",
          }}
        />
        {/* Inner energy */}
        <div
          className={`absolute inset-3 rounded-full ${state === "streaming" ? "animate-[inner-spin_2s_linear_infinite]" : ""}`}
          style={{
            background: `conic-gradient(from 0deg, ${colors.primary}40, ${colors.secondary}40, ${colors.primary}40)`,
            filter: "blur(4px)",
          }}
        />
      </div>

      {/* State label */}
      <div className="absolute -bottom-8 left-1/2 -translate-x-1/2">
        <span
          className="text-[10px] font-mono uppercase tracking-[0.25em] opacity-40 transition-opacity duration-500"
          style={{ color: colors.primary }}
        >
          {state === "idle" ? "awaiting" : state === "listening" ? "listening" : state === "thinking" ? "cognizing" : state === "streaming" ? "manifesting" : "disrupted"}
        </span>
      </div>
    </div>
  );
}
