import { useEffect, useRef } from "react";
import { marked } from "marked";
import hljs from "highlight.js";
import { User, Zap, Paperclip } from "lucide-react";
import type { ChatMessage } from "../../stores/chatStore";

interface StreamCardProps {
  message: ChatMessage;
  depth: number; // 0 = newest (most prominent), higher = older (faded)
  total: number;
}

marked.setOptions({ breaks: true, gfm: true });

/**
 * A single glassmorphism message card floating in the consciousness stream.
 * Depth controls opacity, scale, and blur — older messages recede.
 */
export default function StreamCard({ message, depth, total }: StreamCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isUser = message.role === "user";

  useEffect(() => {
    if (ref.current) {
      ref.current.querySelectorAll("pre code").forEach((el) => {
        hljs.highlightElement(el as HTMLElement);
      });
    }
  }, [message.content]);

  const html = message.content ? (marked.parse(message.content) as string) : "";

  // Depth-based styling: newest = fully visible, oldest = faded and smaller
  const maxVisible = Math.min(total, 8);
  const depthPct = Math.min(depth / maxVisible, 1);
  const opacity = Math.max(1 - depthPct * 0.7, 0.15);
  const scale = Math.max(1 - depthPct * 0.04, 0.88);
  const blur = depth > 5 ? Math.min((depth - 5) * 0.5, 2) : 0;

  return (
    <div
      className={`relative transition-all duration-700 ease-out ${isUser ? "ml-auto mr-4" : "mr-auto ml-4"}`}
      style={{
        opacity,
        transform: `scale(${scale})`,
        filter: blur > 0 ? `blur(${blur}px)` : undefined,
        maxWidth: "min(680px, 85%)",
      }}
    >
      {/* Glass card */}
      <div
        className={`relative rounded-2xl px-5 py-4 backdrop-blur-md border transition-all duration-500 ${
          isUser
            ? "bg-[rgba(139,92,246,0.12)] border-[rgba(139,92,246,0.25)] hover:border-[rgba(139,92,246,0.4)]"
            : "bg-[rgba(0,212,255,0.06)] border-[rgba(0,212,255,0.15)] hover:border-[rgba(0,212,255,0.3)]"
        }`}
        style={{
          boxShadow: isUser
            ? "0 4px 30px rgba(139,92,246,0.08), inset 0 1px 0 rgba(255,255,255,0.05)"
            : "0 4px 30px rgba(0,212,255,0.06), inset 0 1px 0 rgba(255,255,255,0.04)",
        }}
      >
        {/* Role indicator */}
        <div className={`flex items-center gap-2 mb-2 ${isUser ? "justify-end" : ""}`}>
          <div
            className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[9px] font-mono uppercase tracking-[0.2em] ${
              isUser
                ? "bg-[rgba(139,92,246,0.15)] text-[#a78bfa]"
                : "bg-[rgba(0,212,255,0.1)] text-[#67e8f9]"
            }`}
          >
            {isUser ? <User size={9} /> : <Zap size={9} />}
            {isUser ? "you" : "antigravity"}
          </div>
          <span className="text-[9px] font-mono text-[var(--color-text-muted)] opacity-50">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
        </div>

        {/* Content */}
        <div
          ref={ref}
          className="markdown-body text-sm leading-relaxed text-[var(--color-text-primary)]"
          dangerouslySetInnerHTML={{ __html: html }}
        />

        {/* Streaming cursor */}
        {message.streaming && (
          <div className="flex items-center gap-1 mt-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" />
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" style={{ animationDelay: "0.15s" }} />
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" style={{ animationDelay: "0.3s" }} />
          </div>
        )}

        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3">
            {message.attachments.map((a) => (
              <div
                key={a.key}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.08)] text-[10px] font-mono"
              >
                <Paperclip size={10} className="text-[var(--color-accent)] opacity-60" />
                <span className="text-[var(--color-text-secondary)] truncate max-w-[140px]">{a.name}</span>
                <span className="text-[var(--color-text-muted)] opacity-40">
                  {a.size < 1024 ? `${a.size}B` : `${(a.size / 1024).toFixed(0)}K`}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Tool executions */}
        {message.tools && message.tools.length > 0 && (
          <div className="mt-3 space-y-1">
            {message.tools.map((t, i) => (
              <div key={i} className="text-[10px] font-mono px-2.5 py-1.5 rounded-lg bg-[rgba(0,212,255,0.05)] border border-[rgba(0,212,255,0.1)]">
                <span className="text-[var(--color-accent)] opacity-70">{t.name}</span>
                <span className="text-[var(--color-text-muted)] ml-2 opacity-40 truncate">{t.result.slice(0, 80)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Edge glow on newest messages */}
      {depth === 0 && !isUser && (
        <div
          className="absolute -inset-px rounded-2xl pointer-events-none animate-[glow-pulse_3s_ease-in-out_infinite]"
          style={{
            background: "linear-gradient(135deg, rgba(0,212,255,0.1), transparent 40%, transparent 60%, rgba(139,92,246,0.08))",
          }}
        />
      )}
    </div>
  );
}
