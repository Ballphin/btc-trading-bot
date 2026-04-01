import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface Props {
  title: string;
  icon: React.ReactNode;
  content: string;
  defaultOpen?: boolean;
}

export default function AgentReportCard({ title, icon, content, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  if (!content) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass p-4 cursor-pointer"
      onClick={() => setOpen(!open)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent-teal/10 flex items-center justify-center text-accent-teal">
            {icon}
          </div>
          <h3 className="font-semibold text-white text-sm">{title}</h3>
        </div>
        <button aria-label={open ? `Collapse ${title}` : `Expand ${title}`} aria-expanded={open} className="text-slate-400 hover:text-white transition-colors">
          {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-4 pt-4 border-t border-white/5 markdown-content text-sm max-h-96 overflow-y-auto">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
