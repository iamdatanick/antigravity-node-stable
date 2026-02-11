import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { useChatStore } from "../stores/chatStore";
import { useHealth, countServices } from "../api/health";
import { streamChat } from "../api/client";

import ConversationSidebar from "../components/hybrid/ConversationSidebar";
import ModelSelector from "../components/hybrid/ModelSelector";
import MessageStream from "../components/chat/MessageStream";
import NeuralInput from "../components/chat/NeuralInput";
import Core from "../components/chat/Core";
import ParticleField from "../components/chat/ParticleField";

export default function HybridChat() {
  const { messages, isStreaming, addMessage, appendContent, updateMessage, setStreaming } = useChatStore();
  const { data: healthData } = useHealth();
  const [selectedModel, setSelectedModel] = useState("gpt-4o");
  const [isTyping, setIsTyping] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const healthRatio = useMemo(() => {
    const { total, healthy } = countServices(healthData);
    return total > 0 ? healthy / total : 1;
  }, [healthData]);

  const coreState = useMemo(() => {
    if (isStreaming) {
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.content === "") return "thinking";
      return "streaming";
    }
    if (isTyping) return "listening";
    return "idle";
  }, [isStreaming, isTyping, messages]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async (text: string, attachments: any[]) => {
    const userMsgId = addMessage({ role: "user", content: text, attachments });
    const assistantId = addMessage({ role: "assistant", content: "", streaming: true });
    setStreaming(true);

    try {
      abortRef.current = new AbortController();
      for await (const chunk of streamChat([...messages.slice(-6), { role: "user", content: text }], selectedModel, abortRef.current.signal)) {
        appendContent(assistantId, chunk);
      }
    } catch (err) {
      updateMessage(assistantId, { content: "Protocol error: " + (err as Error).message });
    } finally {
      updateMessage(assistantId, { streaming: false });
      setStreaming(false);
    }
  }, [selectedModel, messages, addMessage, appendContent, updateMessage, setStreaming]);

  return (
    <div className="flex h-screen w-full bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] overflow-hidden font-sans antialiased">
      {/* Dynamic Background Layer */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-b from-[var(--color-bg-primary)] via-[var(--color-bg-primary)] to-[#0c1222]" />
        <div className={`absolute inset-0 opacity-20 transition-opacity duration-1000 ${isStreaming ? 'opacity-40' : 'opacity-20'}`}>
          <ParticleField healthRatio={healthRatio} isStreaming={isStreaming} />
        </div>
      </div>

      {/* Library Layer (LibreChat Style) */}
      <ConversationSidebar />

      {/* Orchestration Layer */}
      <div className="flex-1 flex flex-col relative overflow-hidden">
        {/* Command Header */}
        <header className="h-16 border-b border-[var(--color-border)]/50 flex items-center justify-between px-6 bg-[var(--color-bg-primary)]/60 backdrop-blur-xl z-30 shadow-sm">
          <div className="flex items-center gap-4">
             <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-purple)] flex items-center justify-center text-white font-bold shadow-lg shadow-[var(--color-accent)]/10">
              AG
            </div>
            <ModelSelector activeModel={selectedModel} onModelChange={setSelectedModel} />
          </div>
          
          <div className="flex items-center gap-6">
            <div className="h-8 w-[1px] bg-[var(--color-border)]/30" />
            <div className="flex flex-col items-end">
              <div className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full ${healthRatio > 0.9 ? 'bg-[var(--color-green)] shadow-[0_0_8px_var(--color-green)]' : 'bg-[var(--color-yellow)] shadow-[0_0_8px_var(--color-yellow)]'} animate-pulse`}></div>
                <span className="text-[10px] font-mono text-[var(--color-text-muted)] uppercase tracking-[0.2em]">Synchronized</span>
              </div>
              <span className="text-[9px] font-mono opacity-40 uppercase">Region: us-central1-a</span>
            </div>
          </div>
        </header>

        {/* Content & Stream */}
        <main className="flex-1 relative overflow-hidden flex flex-col">
          {/* Ambient Intelligence Core (Goose Style) */}
          <div className={`absolute inset-0 flex items-center justify-center pointer-events-none transition-all duration-1000 ease-out ${messages.length > 0 ? 'opacity-10 scale-[0.35] -translate-y-1/3' : 'opacity-100 scale-100'}`}>
            <Core state={coreState} healthRatio={healthRatio} />
          </div>

          {/* Message Stream with Glass Effect */}
          <div 
            ref={scrollRef}
            className="flex-1 overflow-y-auto z-10 custom-scrollbar scroll-smooth"
          >
            <div className="max-w-4xl mx-auto w-full pt-8 pb-32">
              <MessageStream messages={messages} />
            </div>
          </div>

          {/* Neural Interface Input */}
          <div className="absolute bottom-0 left-0 right-0 p-8 z-20">
             <div className="max-w-4xl mx-auto relative">
               <div className="absolute -inset-4 bg-[var(--color-bg-primary)]/80 backdrop-blur-2xl rounded-[2rem] -z-10 shadow-[0_-20px_50px_rgba(0,0,0,0.5)] border-t border-[var(--color-border)]/20" />
               <NeuralInput 
                onSend={handleSend}
                onStop={() => abortRef.current?.abort()}
                isStreaming={isStreaming}
                onTypingChange={setIsTyping}
              />
              <div className="mt-3 flex items-center justify-between px-2">
                <span className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-[0.2em] opacity-30 font-mono">
                  Antigravity Node v14.1 — Private Inference Mesh
                </span>
                <div className="flex gap-4 opacity-40">
                   <span className="text-[9px] font-mono cursor-help hover:opacity-100">Tokens: {messages.reduce((a, b) => a + b.content.length, 0)}</span>
                   <span className="text-[9px] font-mono cursor-help hover:opacity-100">Latency: 42ms</span>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
