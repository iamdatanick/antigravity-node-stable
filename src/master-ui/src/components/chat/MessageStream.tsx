import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../stores/chatStore";
import StreamCard from "./StreamCard";

interface MessageStreamProps {
  messages: ChatMessage[];
}

/**
 * The message stream — floating glass cards arranged with depth perspective.
 * Newest messages at the bottom, fully opaque. Older messages fade upward.
 */
export default function MessageStream({ messages }: MessageStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  if (messages.length === 0) return null;

  return (
    <div
      ref={scrollRef}
      className="absolute inset-x-0 bottom-28 top-0 overflow-y-auto overflow-x-hidden px-4 pb-6 pt-20"
      style={{
        zIndex: 2,
        maskImage: "linear-gradient(to bottom, transparent 0%, black 8%, black 85%, transparent 100%)",
        WebkitMaskImage: "linear-gradient(to bottom, transparent 0%, black 8%, black 85%, transparent 100%)",
      }}
    >
      <div className="flex flex-col gap-4 max-w-3xl mx-auto">
        {messages.map((msg, i) => (
          <StreamCard
            key={msg.id}
            message={msg}
            depth={messages.length - 1 - i}
            total={messages.length}
          />
        ))}
      </div>
    </div>
  );
}
