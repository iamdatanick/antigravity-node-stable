import { MessageSquare, Plus, History, Trash2, Search, FolderPlus } from "lucide-react";
import { useChatStore } from "../../stores/chatStore";

export default function ConversationSidebar() {
  const { clear } = useChatStore();
  
  return (
    <div className="flex flex-col h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)] w-64 shrink-0 overflow-hidden">
      <div className="p-4 flex flex-col gap-2">
        <button 
          onClick={clear}
          className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg border border-[rgba(255,255,255,0.1)] hover:bg-[var(--color-bg-tertiary)] transition-all text-sm font-medium group"
        >
          <Plus size={16} className="text-[var(--color-accent)] group-hover:rotate-90 transition-transform" />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 custom-scrollbar">
        <div className="px-3 py-2 flex items-center justify-between group">
          <span className="text-[10px] font-bold text-[var(--color-text-muted)] uppercase tracking-widest">Recent History</span>
          <FolderPlus size={12} className="text-[var(--color-text-muted)] opacity-0 group-hover:opacity-100 cursor-pointer hover:text-[var(--color-accent)] transition-all" />
        </div>
        
        <div className="space-y-0.5">
          <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg bg-[var(--color-accent-dim)] text-[var(--color-accent)] text-sm text-left border border-[var(--color-accent)]/20">
            <MessageSquare size={14} />
            <span className="truncate font-medium">Phoenix Reconstruction...</span>
          </button>
          
          {[1, 2, 3].map((i) => (
            <button key={i} className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)] text-sm text-left transition-colors group">
              <History size={14} className="opacity-40 group-hover:opacity-100 transition-opacity" />
              <span className="truncate flex-1">Analysis of System Logs {i}</span>
              <Trash2 size={12} className="opacity-0 group-hover:opacity-60 hover:text-[var(--color-red)] transition-all" />
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 border-t border-[var(--color-border)] bg-[var(--color-bg-primary)]/50">
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)] cursor-pointer transition-colors">
          <Search size={16} />
          <span className="text-sm">Search threads...</span>
        </div>
      </div>
    </div>
  );
}
