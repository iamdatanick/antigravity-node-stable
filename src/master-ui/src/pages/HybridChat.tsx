
import React, { useState, useEffect } from 'react';
import { ConversationSidebar } from '../components/hybrid/ConversationSidebar';
import { ModelSelector } from '../components/hybrid/ModelSelector';
import { ChatBubble } from '../components/chat/ChatBubble';
import { useChatStore } from '../stores/chatStore';

export const HybridChat: React.FC = () => {
  const { messages, sendMessage, isStreaming } = useChatStore();
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex h-full bg-[var(--color-bg-primary)] overflow-hidden">
      <ConversationSidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b border-[var(--color-border-primary)] flex items-center justify-between px-6 bg-[var(--color-bg-secondary)]">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Phoenix Hybrid</h2>
            <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
          </div>
          <ModelSelector />
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
          {messages.map((m, i) => (
            <ChatBubble key={m.id || i} message={m} />
          ))}
          {isStreaming && (
            <div className="flex items-center space-x-2 text-[var(--color-text-tertiary)] animate-pulse">
              <div className="w-2 h-2 rounded-full bg-current"></div>
              <span>Phoenix is thinking...</span>
            </div>
          )}
        </div>

        <footer className="p-6 bg-[var(--color-bg-secondary)] border-t border-[var(--color-border-primary)]">
          <div className="max-w-4xl mx-auto flex items-end space-x-4">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
              placeholder="Message Phoenix..."
              className="flex-1 min-h-[48px] max-h-48 p-3 rounded-xl bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] text-[var(--color-text-primary)] focus:ring-2 focus:ring-[var(--color-accent)] outline-none resize-none transition-all"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="h-12 w-12 rounded-xl bg-[var(--color-accent)] text-white flex items-center justify-center hover:opacity-90 disabled:opacity-50 transition-all"
            >
              <svg className="w-5 h-5 rotate-90" fill="currentColor" viewBox="0 0 20 20"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" /></svg>
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};
