import { useState, useRef, useCallback } from "react";
import { Send, Paperclip, StopCircle, X, FileText, Loader2 } from "lucide-react";
import { uploadFile } from "../../api/client";
import type { Attachment } from "../../stores/chatStore";

interface ChatInputProps {
  onSend: (text: string, attachments: Attachment[]) => void;
  onStop: () => void;
  isStreaming: boolean;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const TEXT_TYPES = [
  "text/", "application/json", "application/xml", "application/javascript",
  "application/typescript", "application/x-yaml", "application/toml",
  "application/sql", "application/x-sh",
];

function isTextFile(file: File): boolean {
  if (TEXT_TYPES.some((t) => file.type.startsWith(t))) return true;
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return ["txt", "md", "csv", "json", "yaml", "yml", "xml", "html", "css", "js", "ts",
    "tsx", "jsx", "py", "rs", "go", "java", "sql", "sh", "ps1", "toml", "ini", "cfg",
    "log", "env", "dockerfile"].includes(ext);
}

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

export default function ChatInput({ onSend, onStop, isStreaming }: ChatInputProps) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if ((!trimmed && attachments.length === 0) || isStreaming) return;
    onSend(trimmed, attachments);
    setText("");
    setAttachments([]);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [text, attachments, isStreaming, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 200) + "px";
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        // Read text content client-side for text files under 5MB
        let content: string | undefined;
        if (isTextFile(file) && file.size <= 5 * 1024 * 1024) {
          content = await readFileText(file);
        }

        // Upload to S3 for persistence
        const result = await uploadFile(file);

        setAttachments((prev) => [
          ...prev,
          {
            name: file.name,
            size: file.size,
            key: result.key,
            type: file.type || "application/octet-stream",
            content,
          },
        ]);
      }
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
      // Reset input so the same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const removeAttachment = (key: string) => {
    setAttachments((prev) => prev.filter((a) => a.key !== key));
  };

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-4">
      {/* Attachment chips */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3 max-w-4xl mx-auto">
          {attachments.map((a) => (
            <div
              key={a.key}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] text-xs"
            >
              <FileText size={14} className="text-[var(--color-accent)] shrink-0" />
              <span className="text-[var(--color-text-primary)] truncate max-w-[200px]">{a.name}</span>
              <span className="text-[var(--color-text-muted)]">{formatSize(a.size)}</span>
              <button
                onClick={() => removeAttachment(a.key)}
                className="text-[var(--color-text-muted)] hover:text-[var(--color-red)] transition-colors"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-end gap-3 max-w-4xl mx-auto">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />

        {/* File upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading || isStreaming}
          className="p-2 text-[var(--color-text-muted)] hover:text-[var(--color-accent)] transition-colors rounded-lg hover:bg-[var(--color-bg-tertiary)] disabled:opacity-40 disabled:cursor-not-allowed"
          title="Attach file"
        >
          {uploading ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <Paperclip size={18} />
          )}
        </button>

        {/* Textarea */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder={attachments.length > 0 ? "Add a message about the file(s)..." : "Ask anything... (Shift+Enter for new line)"}
            rows={1}
            className="w-full resize-none bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] border border-[var(--color-border)] rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-[var(--color-accent)] transition-colors"
          />
        </div>

        {/* Send / Stop button */}
        {isStreaming ? (
          <button
            onClick={onStop}
            className="p-2.5 rounded-xl bg-[var(--color-red)] text-white hover:opacity-90 transition-opacity"
            title="Stop generation"
          >
            <StopCircle size={18} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!text.trim() && attachments.length === 0}
            className="p-2.5 rounded-xl bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-purple)] text-white hover:opacity-90 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
            title="Send message"
          >
            <Send size={18} />
          </button>
        )}
      </div>
    </div>
  );
}
