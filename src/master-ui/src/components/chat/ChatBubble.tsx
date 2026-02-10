import { useEffect, useRef } from "react";
import { marked } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/tokyo-night-dark.css";
import { Bot, User, ChevronDown, ChevronRight, Wrench } from "lucide-react";
import { useState } from "react";
import type { ChatMessage } from "../../stores/chatStore";

marked.setOptions({
  breaks: true,
  gfm: true,
});

function ToolPanel({ tools }: { tools: { name: string; result: string }[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 w-full px-3 py-2 text-xs text-[var(--color-text-muted)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
      >
        <Wrench size={12} />
        <span>{tools.length} tool{tools.length > 1 ? "s" : ""} executed</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && (
        <div className="border-t border-[var(--color-border)] p-3 space-y-2">
          {tools.map((t, i) => (
            <div key={i} className="text-xs">
              <div className="font-mono text-[var(--color-accent)] mb-1">{t.name}</div>
              <pre className="bg-[var(--color-bg-secondary)] p-2 rounded text-[var(--color-text-secondary)] overflow-x-auto whitespace-pre-wrap">
                {t.result}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ChatBubble({ message }: { message: ChatMessage }) {
  const ref = useRef<HTMLDivElement>(null);
  const isUser = message.role === "user";

  useEffect(() => {
    if (ref.current) {
      ref.current.querySelectorAll("pre code").forEach((el) => {
        hljs.highlightElement(el as HTMLElement);
      });
    }
  }, [message.content]);

  const html = marked.parse(message.content) as string;

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
          isUser
            ? "bg-[var(--color-purple-dim)] text-[var(--color-purple)]"
            : "bg-[var(--color-accent-dim)] text-[var(--color-accent)]"
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Bubble */}
      <div className={`max-w-[75%] min-w-0 ${isUser ? "items-end" : ""}`}>
        {/* Thinking indicator */}
        {message.thinking && message.thinking.length > 0 && (
          <div className="mb-2 space-y-1">
            {message.thinking.map((step, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-[11px] text-[var(--color-text-muted)] italic"
              >
                <span className="w-1 h-1 rounded-full bg-[var(--color-purple)] animate-pulse" />
                {step}
              </div>
            ))}
          </div>
        )}

        <div
          ref={ref}
          className={`markdown-body rounded-xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-[var(--color-purple)] text-white rounded-tr-sm"
              : "bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)] border border-[var(--color-border)] rounded-tl-sm"
          }`}
          dangerouslySetInnerHTML={{ __html: html }}
        />

        {/* Streaming cursor */}
        {message.streaming && (
          <span className="inline-block w-2 h-4 bg-[var(--color-accent)] animate-pulse rounded-sm ml-1" />
        )}

        {/* Tool executions */}
        {message.tools && message.tools.length > 0 && <ToolPanel tools={message.tools} />}

        {/* Timestamp */}
        <div
          className={`text-[10px] text-[var(--color-text-muted)] mt-1 ${isUser ? "text-right" : ""}`}
        >
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
