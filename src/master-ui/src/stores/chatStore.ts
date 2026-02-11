import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Attachment {
  name: string;
  size: number;
  key: string;
  type: string;
  content?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  thinking?: string[];
  tools?: { name: string; result: string }[];
  streaming?: boolean;
  attachments?: Attachment[];
}

interface ChatState {
  messages: ChatMessage[];
  isStreaming: boolean;
  addMessage: (msg: Omit<ChatMessage, "id" | "timestamp">) => string;
  updateMessage: (id: string, partial: Partial<ChatMessage>) => void;
  appendContent: (id: string, chunk: string) => void;
  setStreaming: (v: boolean) => void;
  clear: () => void;
}

let msgId = Date.now();

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      messages: [],
      isStreaming: false,
      addMessage: (msg) => {
        const id = `msg-${++msgId}`;
        set((s) => ({
          messages: [...s.messages, { ...msg, id, timestamp: Date.now() }],
        }));
        return id;
      },
      updateMessage: (id, partial) =>
        set((s) => ({
          messages: s.messages.map((m) => (m.id === id ? { ...m, ...partial } : m)),
        })),
      appendContent: (id, chunk) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id ? { ...m, content: m.content + chunk } : m,
          ),
        })),
      setStreaming: (isStreaming) => set({ isStreaming }),
      clear: () => {
        msgId = Date.now();
        set({ messages: [], isStreaming: false });
      },
    }),
    {
      name: "antigravity-chat",
      partialize: (state) => ({
        messages: state.messages.filter((m) => !m.streaming),
      }),
    }
  )
);
