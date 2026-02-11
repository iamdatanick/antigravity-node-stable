import { Construction } from "lucide-react";

interface UnavailableProps {
  feature: string;
  reason: string;
}

export default function Unavailable({ feature, reason }: UnavailableProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
      <Construction size={48} className="text-[var(--color-text-muted)] opacity-40" />
      <h1 className="text-lg font-bold text-[var(--color-text-primary)]">{feature}</h1>
      <p className="text-sm text-[var(--color-text-muted)] max-w-md">{reason}</p>
    </div>
  );
}
