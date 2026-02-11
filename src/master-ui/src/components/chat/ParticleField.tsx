import { useEffect, useRef, useCallback } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  hue: number;
  life: number;
  maxLife: number;
}

interface ParticleFieldProps {
  healthRatio: number; // 0 = all degraded, 1 = all healthy
  isStreaming: boolean;
}

const PARTICLE_COUNT = 120;
const CONNECTION_DIST = 140;

export default function ParticleField({ healthRatio, isStreaming }: ParticleFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);
  const healthRef = useRef(healthRatio);
  const streamRef = useRef(isStreaming);

  healthRef.current = healthRatio;
  streamRef.current = isStreaming;

  const createParticle = useCallback((w: number, h: number): Particle => {
    const maxLife = 400 + Math.random() * 600;
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      radius: 1 + Math.random() * 2,
      alpha: 0.1 + Math.random() * 0.4,
      hue: 195 + Math.random() * 30, // cyan range
      life: Math.random() * maxLife,
      maxLife,
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
      ctx.scale(dpr, dpr);
    };
    resize();
    window.addEventListener("resize", resize);

    // Init particles
    const w = window.innerWidth;
    const h = window.innerHeight;
    particlesRef.current = Array.from({ length: PARTICLE_COUNT }, () =>
      createParticle(w, h),
    );

    const animate = () => {
      const W = window.innerWidth;
      const H = window.innerHeight;
      const hr = healthRef.current;
      const streaming = streamRef.current;

      ctx.clearRect(0, 0, W, H);

      const particles = particlesRef.current;

      // Health-reactive base hue: 195 (cyan) when healthy, shifts to 30 (amber) when degraded
      const baseHue = 195 * hr + 30 * (1 - hr);
      const speed = streaming ? 1.8 : 0.6;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.life++;
        if (p.life > p.maxLife) {
          particles[i] = createParticle(W, H);
          continue;
        }

        // Fade in/out based on life
        const lifePct = p.life / p.maxLife;
        const fade = lifePct < 0.1 ? lifePct / 0.1 : lifePct > 0.85 ? (1 - lifePct) / 0.15 : 1;

        p.x += p.vx * speed;
        p.y += p.vy * speed;

        // Wrap around
        if (p.x < -10) p.x = W + 10;
        if (p.x > W + 10) p.x = -10;
        if (p.y < -10) p.y = H + 10;
        if (p.y > H + 10) p.y = -10;

        // Subtle drift toward center when streaming
        if (streaming) {
          const dx = W / 2 - p.x;
          const dy = H * 0.38 - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 50) {
            p.vx += (dx / dist) * 0.003;
            p.vy += (dy / dist) * 0.003;
          }
        }

        // Dampen velocity
        p.vx *= 0.999;
        p.vy *= 0.999;

        // Draw particle
        const hue = baseHue + (p.hue - 195);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${hue}, 80%, 65%, ${p.alpha * fade})`;
        ctx.fill();

        // Draw connections
        for (let j = i + 1; j < particles.length; j++) {
          const q = particles[j];
          const dx = p.x - q.x;
          const dy = p.y - q.y;
          const dist = dx * dx + dy * dy;
          if (dist < CONNECTION_DIST * CONNECTION_DIST) {
            const strength = 1 - Math.sqrt(dist) / CONNECTION_DIST;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = `hsla(${hue}, 60%, 50%, ${strength * 0.12 * fade})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [createParticle]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
