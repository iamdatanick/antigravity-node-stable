import { useRef, useCallback, useState, useMemo } from "react";
import { useChatStore } from "../stores/chatStore";
import type { Attachment } from "../stores/chatStore";
import { streamChat } from "../api/client";
import { useModels } from "../api/models";
import { useHealth, countServices } from "../api/health";

import ParticleField from "../components/chat/ParticleField";
import Core from "../components/chat/Core";
import MessageStream from "../components/chat/MessageStream";
import NeuralInput from "../components/chat/NeuralInput";
import AmbientHUD from "../components/chat/AmbientHUD";

/**
 * Consciousness Chamber — Antigravity Node v14.1
 *
 * A full-viewport AI interface where the AI exists as a living entity.
 * Three Z-layers: Aether (particles) → Core (AI orb) → Stream (messages).
 */
export default function Chat() {
  const { messages, isStreaming, addMessage, appendContent, updateMessage, setStreaming, clear } =
    useChatStore();
  const abortRef = useRef<AbortController | null>(null);
  const { data: modelsData } = useModels();
  const { data: healthData } = useHealth();
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isTyping, setIsTyping] = useState(false);

  // Resolve active model
  const models = modelsData?.data ?? [];
  const activeModel = selectedModel || models[0]?.id || "gpt-4o";

  // Compute health ratio (0-1) for ambient visuals
  const healthRatio = useMemo(() => {
    const { total, healthy } = countServices(healthData);
    return total > 0 ? healthy / total : 1; // Default to healthy when no data
  }, [healthData]);

  // Derive AI core state from conversation state
  const coreState = useMemo((): "idle" | "listening" | "thinking" | "streaming" | "error" => {
    if (isStreaming) {
      // Check if the latest assistant message has content yet
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.content === "") return "thinking";
      return "streaming";
    }
    if (isTyping) return "listening";
    return "idle";
  }, [isStreaming, isTyping, messages]);

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      // Build display content
      let displayContent = text;
      if (attachments.length > 0 && !text) {
        displayContent = `Sent ${attachments.length} file${attachments.length > 1 ? "s" : ""}: ${attachments.map((a) => a.name).join(", ")}`;
      }

      addMessage({
        role: "user",
        content: displayContent,
        attachments: attachments.length > 0 ? attachments : undefined,
      });
      const assistantId = addMessage({ role: "assistant", content: "", streaming: true });
      setStreaming(true);

      // Build LLM messages with file context injection
      const fileContext = attachments
        .filter((a) => a.content)
        .map((a) => `<file name="${a.name}" type="${a.type}">\n${a.content}\n</file>`)
        .join("\n\n");

      const fileRefs = attachments
        .filter((a) => !a.content)
        .map((a) => `[Binary file: ${a.name} (${(a.size / 1024).toFixed(1)} KB) — stored at ${a.key}]`)
        .join("\n");

      let enrichedText = text;
      if (fileContext || fileRefs) {
        const parts: string[] = [];
        if (fileContext) parts.push(`The user attached the following file(s):\n\n${fileContext}`);
        if (fileRefs) parts.push(`The user also attached binary file(s):\n${fileRefs}`);
        if (text) parts.push(`User message: ${text}`);
        else parts.push("The user sent these files without a message. Acknowledge the files and offer to help analyze them.");
        enrichedText = parts.join("\n\n");
      }

      const chatMessages = [
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        { role: "user", content: enrichedText },
      ];

      try {
        abortRef.current = new AbortController();
        for await (const chunk of streamChat(chatMessages, activeModel, abortRef.current.signal)) {
          appendContent(assistantId, chunk);
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          appendContent(assistantId, `\n\n_Error: ${(err as Error).message}_`);
        }
      } finally {
        updateMessage(assistantId, { streaming: false });
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [messages, addMessage, appendContent, updateMessage, setStreaming, activeModel],
  );

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden bg-[var(--color-bg-primary)]" style={{ animation: "consciousness-fade 0.8s ease-out" }}>
      {/* Layer 0: Aether — health-reactive particle field */}
      <ParticleField healthRatio={healthRatio} isStreaming={isStreaming} />

      {/* Layer 1: Core — AI entity (centered, behind messages when scrolled) */}
      {messages.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ zIndex: 1 }}>
          <Core state={coreState} healthRatio={healthRatio} />

          {/* Welcome state — prompt suggestions */}
          <div className="mt-20 flex flex-col items-center gap-4 animate-[float-in_0.8s_ease-out_0.3s_both]">
            <h1
              className="text-lg font-light tracking-[0.3em] uppercase text-[var(--color-text-muted)] opacity-40"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              Antigravity
            </h1>
            <div className="flex flex-wrap gap-2 justify-center max-w-md mt-2">
              {[
                "System status",
                "Available models",
                "Budget remaining",
                "What can you do?",
              ].map((q, i) => (
                <button
                  key={q}
                  onClick={() => handleSend(q, [])}
                  className="px-3 py-1.5 rounded-xl text-[10px] font-mono tracking-wide border border-[rgba(255,255,255,0.06)] bg-[rgba(255,255,255,0.02)] text-[var(--color-text-muted)] hover:border-[rgba(0,212,255,0.25)] hover:text-[var(--color-accent)] hover:bg-[rgba(0,212,255,0.04)] transition-all duration-300 backdrop-blur-sm"
                  style={{ animationDelay: `${0.4 + i * 0.1}s` }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Layer 1b: Core — miniaturized when conversation is active */}
      {messages.length > 0 && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2" style={{ zIndex: 3, transform: "translateX(-50%) scale(0.35)" }}>
          <Core state={coreState} healthRatio={healthRatio} />
        </div>
      )}

      {/* Layer 2: Message Stream */}
      <MessageStream messages={messages} />

      {/* Layer 3: Neural Input */}
      <NeuralInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming}
        onTypingChange={setIsTyping}
      />

      {/* Layer 4: Ambient HUD overlays */}
      <AmbientHUD
        activeModel={activeModel}
        onModelChange={setSelectedModel}
        isStreaming={isStreaming}
      />

      {/* Clear conversation — tiny button top-center */}
      {messages.length > 0 && (
        <button
          onClick={clear}
          className="fixed top-4 left-1/2 -translate-x-1/2 px-3 py-1 rounded-xl text-[9px] font-mono tracking-[0.15em] text-[var(--color-text-muted)] opacity-20 hover:opacity-60 bg-[rgba(15,23,42,0.7)] border border-[rgba(255,255,255,0.04)] backdrop-blur-md transition-all duration-300 hover:border-[rgba(239,68,68,0.2)] hover:text-[var(--color-red)]"
          style={{ zIndex: 20 }}
        >
          CLEAR
        </button>
      )}
    </div>
  );
}
