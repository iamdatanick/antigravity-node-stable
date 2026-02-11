import { ChevronDown, Cpu, Sparkles, Zap, ShieldCheck, Globe, FlaskConical } from "lucide-react";
import { useState, useEffect } from "react";
import { apiFetch } from "../../api/client";

interface ModelSelectorProps {
  activeModel: string;
  onModelChange: (id: string) => void;
}

export default function ModelSelector({ activeModel, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<any[]>([]);

  useEffect(() => {
    apiFetch<{ data: any[] }>("/v1/models")
      .then(res => setModels(res.data))
      .catch(() => setModels([
        { id: "gpt-4o", owned_by: "antigravity" },
        { id: "gpt-4o-mini", owned_by: "antigravity" },
        { id: "tinyllama", owned_by: "local" }
      ]));
  }, []);

  const getIcon = (id: string) => {
    if (id.includes("gpt-4")) return <Sparkles size={14} className="text-purple-400" />;
    if (id.includes("tinyllama")) return <Cpu size={14} className="text-blue-400" />;
    if (id.includes("gemini") || id.includes("vertex")) return <ShieldCheck size={14} className="text-green-400" />;
    if (id.includes("claude")) return <Zap size={14} className="text-amber-400" />;
    return <Globe size={14} className="text-slate-400" />;
  };

  return (
    <div className="relative">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-bg-tertiary)]/50 border border-[var(--color-border)] hover:border-[var(--color-accent)] transition-all min-w-[200px] backdrop-blur-sm shadow-inner"
      >
        <div className="w-5 h-5 rounded flex items-center justify-center bg-[var(--color-bg-primary)] border border-[var(--color-border)]">
          {getIcon(activeModel)}
        </div>
        <span className="text-[10px] font-mono uppercase tracking-widest truncate flex-1 text-left">
          {activeModel}
        </span>
        <ChevronDown size={14} className={`text-[var(--color-text-muted)] transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-2 w-72 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 backdrop-blur-xl">
            <div className="px-4 py-2.5 text-[9px] font-bold text-[var(--color-text-muted)] uppercase tracking-[0.2em] border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]/50 flex items-center justify-between">
              <span>Standard Access</span>
              <FlaskConical size={10} />
            </div>
            
            <div className="max-h-64 overflow-y-auto py-1 custom-scrollbar">
              {models.filter(m => m.owned_by !== 'vertex').map((m) => (
                <button
                  key={m.id}
                  onClick={() => { onModelChange(m.id); setIsOpen(false); }}
                  className={`flex items-center gap-3 w-full px-4 py-2.5 text-xs text-left hover:bg-[var(--color-bg-tertiary)] transition-colors ${activeModel === m.id ? 'text-[var(--color-accent)] bg-[var(--color-accent-dim)]' : 'text-[var(--color-text-secondary)]'}`}
                >
                  <div className="w-6 h-6 rounded flex items-center justify-center bg-[var(--color-bg-primary)]/50 border border-[var(--color-border)]">
                    {getIcon(m.id)}
                  </div>
                  <div className="flex flex-col">
                    <span className="font-medium tracking-tight">{m.id}</span>
                    <span className="text-[9px] opacity-40 uppercase">{m.owned_by}</span>
                  </div>
                </button>
              ))}

              <div className="px-4 py-2.5 text-[9px] font-bold text-[var(--color-text-muted)] uppercase tracking-[0.2em] border-y border-[var(--color-border)] bg-[var(--color-bg-primary)]/50 flex items-center justify-between">
                <span>Vertex Agent Garden</span>
                <ShieldCheck size={10} className="text-[var(--color-green)]" />
              </div>
              
              {/* Vertex Models wired to back end */}
              {['gemini-1.5-pro', 'gemini-1.5-flash', 'vertex-medlm-large'].map(v => (
                <button
                  key={v}
                  onClick={() => { onModelChange(`vertex/${v}`); setIsOpen(false); }}
                  className={`flex items-center gap-3 w-full px-4 py-2.5 text-xs text-left hover:bg-[var(--color-bg-tertiary)] transition-colors ${activeModel === `vertex/${v}` ? 'text-[var(--color-accent)] bg-[var(--color-accent-dim)]' : 'text-[var(--color-text-secondary)]'}`}
                >
                  <div className="w-6 h-6 rounded flex items-center justify-center bg-[var(--color-green-dim)]/20 border border-[var(--color-green)]/30">
                    <ShieldCheck size={14} className="text-[var(--color-green)]" />
                  </div>
                  <div className="flex flex-col">
                    <span className="font-medium tracking-tight text-[var(--color-text-primary)]">{v}</span>
                    <span className="text-[9px] text-[var(--color-green)] uppercase opacity-60 font-mono">Agent Garden</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
