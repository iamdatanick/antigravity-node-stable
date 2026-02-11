import { useEffect, useRef, useCallback, useState } from "react";
import { useChatStore } from "../stores/chatStore";
import { streamChat } from "../api/client";
import { useModels } from "../api/models";
import ChatBubble from "../components/chat/ChatBubble";
import ChatInput from "../components/chat/ChatInput";
import { Bot, ChevronDown, Trash2 } from "lucide-react";

export default function Chat() {
  const { messages, isStreaming, addMessage, appendContent, updateMessage, setStreaming, clear } =
    useChatStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { data: modelsData } = useModels();
  const [selectedModel, setSelectedModel] = useState<string>("");

  // Resolve active model: user selection > first available > fallback
  const models = modelsData?.data ?? [];
  const activeModel = selectedModel || models[0]?.id || "gpt-4o";

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(
    async (text: string) => {
      addMessage({ role: "user", content: text });
      const assistantId = addMessage({ role: "assistant", content: "", streaming: true });
      setStreaming(true);

      const chatMessages = [
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        { role: "user", content: text },
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
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Chat</h1>
            <p className="text-sm text-[var(--color-text-muted)]">Talk to Antigravity Node</p>
          </div>
          <div className="relative">
            <select
              value={activeModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isStreaming}
              className="appearance-none pl-3 pr-8 py-1.5 text-xs font-mono rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)] hover:border-[var(--color-accent)] transition-colors disabled:opacity-50 cursor-pointer"
            >
              {models.length > 0 ? (
                models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} ({m.owned_by})
                  </option>
                ))
              ) : (
                <option value="gpt-4o">gpt-4o (fallback)</option>
              )}
            </select>
            <ChevronDown
              size={12}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)] pointer-events-none"
            />
          </div>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clear}
            className="flex items-center gap-1.5 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-red)] transition-colors px-3 py-1.5 rounded-lg hover:bg-[var(--color-red-dim)]"
          >
            <Trash2 size={12} />
            Clear
          </button>
        )}
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-purple)] flex items-center justify-center">
              <Bot size={32} className="text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
                Antigravity Chat
              </h2>
              <p className="text-sm text-[var(--color-text-muted)] max-w-md mt-1">
                Ask questions, run queries, analyze data, or get help with the system.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {[
                "What's the system status?",
                "Show me recent memory traces",
                "How much budget is remaining?",
                "List available tools",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="text-xs px-3 py-1.5 rounded-lg border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <ChatBubble key={msg.id} message={msg} />)
        )}
      </div>

      <ChatInput onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} />
    </div>
  );
}
