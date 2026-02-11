import { useState, useRef, useCallback, useEffect } from "react";
import { ArrowUp, Square, Paperclip, Loader2, X, FileText } from "lucide-react";
import { uploadFile } from "../../api/client";
import type { Attachment } from "../../stores/chatStore";

interface NeuralInputProps {
  onSend: (text: string, attachments: Attachment[]) => void;
  onStop: () => void;
  isStreaming: boolean;
  onTypingChange?: (isTyping: boolean) => void;
}

const TEXT_EXTENSIONS = [
  "txt", "md", "csv", "json", "yaml", "yml", "xml", "html", "css", "js", "ts",
  "tsx", "jsx", "py", "rs", "go", "java", "sql", "sh", "ps1", "toml", "ini",
  "cfg", "log", "env", "dockerfile",
];

function isTextFile(file: File): boolean {
  if (file.type.startsWith("text/")) return true;
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return TEXT_EXTENSIONS.includes(ext);
}

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}K`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}M`;
}

export default function NeuralInput({ onSend, onStop, isStreaming, onTypingChange }: NeuralInputProps) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Notify parent about typing state
  useEffect(() => {
    onTypingChange?.(text.length > 0);
  }, [text, onTypingChange]);

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
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        let content: string | undefined;
        if (isTextFile(file) && file.size <= 5 * 1024 * 1024) {
          content = await readFileText(file);
        }
        const result = await uploadFile(file);
        setAttachments((prev) => [
          ...prev,
          { name: file.name, size: file.size, key: result.key, type: file.type || "application/octet-stream", content },
        ]);
      }
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const hasContent = text.trim().length > 0 || attachments.length > 0;

  return (
    <div
      className="absolute bottom-0 inset-x-0 px-4 pb-5 pt-3"
      style={{
        zIndex: 10,
        background: "linear-gradient(to top, rgba(15,23,42,0.95) 0%, rgba(15,23,42,0.7) 60%, transparent 100%)",
      }}
    >
      {/* Attachment chips */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3 max-w-2xl mx-auto">
          {attachments.map((a) => (
            <div
              key={a.key}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-[rgba(139,92,246,0.1)] border border-[rgba(139,92,246,0.2)] text-[10px] font-mono backdrop-blur-sm"
            >
              <FileText size={11} className="text-[var(--color-purple)] opacity-70" />
              <span className="text-[var(--color-text-secondary)] truncate max-w-[150px]">{a.name}</span>
              <span className="text-[var(--color-text-muted)] opacity-50">{formatSize(a.size)}</span>
              <button onClick={() => setAttachments((p) => p.filter((x) => x.key !== a.key))} className="text-[var(--color-text-muted)] hover:text-[var(--color-red)] transition-colors ml-0.5">
                <X size={10} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="max-w-2xl mx-auto">
        <div
          className={`relative flex items-end gap-2 rounded-2xl border backdrop-blur-xl transition-all duration-300 ${
            focused
              ? "border-[rgba(0,212,255,0.3)] bg-[rgba(0,212,255,0.04)] shadow-[0_0_30px_rgba(0,212,255,0.08)]"
              : "border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)]"
          }`}
        >
          {/* Hidden file input */}
          <input ref={fileInputRef} type="file" multiple className="hidden" onChange={handleFileSelect} />

          {/* Attach button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || isStreaming}
            className="p-3 pb-3.5 text-[var(--color-text-muted)] hover:text-[var(--color-accent)] transition-colors disabled:opacity-30"
          >
            {uploading ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
          </button>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Think something..."
            rows={1}
            className="flex-1 resize-none bg-transparent text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] placeholder:opacity-40 py-3.5 text-sm leading-relaxed focus:outline-none"
            style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", fontSize: "13px" }}
          />

          {/* Send / Stop */}
          {isStreaming ? (
            <button
              onClick={onStop}
              className="m-2 p-2 rounded-xl bg-[var(--color-red)] text-white hover:opacity-80 transition-opacity"
            >
              <Square size={14} />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!hasContent}
              className={`m-2 p-2 rounded-xl transition-all duration-300 ${
                hasContent
                  ? "bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-purple)] text-white shadow-[0_0_20px_rgba(0,212,255,0.2)] hover:shadow-[0_0_30px_rgba(0,212,255,0.35)]"
                  : "bg-[rgba(255,255,255,0.05)] text-[var(--color-text-muted)] opacity-30 cursor-not-allowed"
              }`}
            >
              <ArrowUp size={14} />
            </button>
          )}
        </div>

        {/* Ambient hint */}
        <div className="flex items-center justify-center mt-2 gap-3">
          <span className="text-[9px] font-mono text-[var(--color-text-muted)] opacity-25 tracking-[0.15em]">
            SHIFT+ENTER for new line
          </span>
        </div>
      </div>
    </div>
  );
}
