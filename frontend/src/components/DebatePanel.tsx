import { useState } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { clsx } from 'clsx';

interface Props {
  type: 'investment' | 'risk';
  data: {
    bull_history?: string;
    bear_history?: string;
    judge_decision?: string;
    aggressive_history?: string;
    conservative_history?: string;
    neutral_history?: string;
  };
}

export default function DebatePanel({ type, data }: Props) {
  const isInvestment = type === 'investment';
  const tabs = isInvestment
    ? [
        { key: 'bull', label: 'Bull', color: 'text-emerald-400', content: data.bull_history },
        { key: 'bear', label: 'Bear', color: 'text-red-400', content: data.bear_history },
        { key: 'judge', label: 'Judge', color: 'text-amber-400', content: data.judge_decision },
      ]
    : [
        { key: 'aggressive', label: 'Aggressive', color: 'text-red-400', content: data.aggressive_history },
        { key: 'conservative', label: 'Conservative', color: 'text-blue-400', content: data.conservative_history },
        { key: 'neutral', label: 'Neutral', color: 'text-slate-300', content: data.neutral_history },
        { key: 'judge', label: 'Judge', color: 'text-amber-400', content: data.judge_decision },
      ];

  const [active, setActive] = useState(tabs[0].key);
  const activeTab = tabs.find(t => t.key === active);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass overflow-hidden"
    >
      <div role="tablist" aria-label={`${isInvestment ? 'Investment' : 'Risk'} debate perspectives`} className="flex border-b border-white/5">
        {tabs.map(tab => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={active === tab.key}
            onClick={() => setActive(tab.key)}
            className={clsx(
              'flex-1 px-4 py-3 min-h-[44px] text-sm font-medium transition-colors relative',
              active === tab.key ? tab.color : 'text-slate-500 hover:text-slate-300',
            )}
          >
            {tab.label}
            {active === tab.key && (
              <motion.div
                layoutId={`debate-${type}-indicator`}
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-current"
              />
            )}
          </button>
        ))}
      </div>
      <div className="p-4 max-h-96 overflow-y-auto markdown-content text-sm">
        {activeTab?.content ? (
          <ReactMarkdown>{activeTab.content}</ReactMarkdown>
        ) : (
          <p className="text-slate-500 italic">No data available</p>
        )}
      </div>
    </motion.div>
  );
}
