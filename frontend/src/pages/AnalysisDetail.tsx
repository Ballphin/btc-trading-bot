import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, BarChart3, MessageSquare, Newspaper, PieChart } from 'lucide-react';
import SignalBadge from '../components/SignalBadge';
import AgentReportCard from '../components/AgentReportCard';
import DebatePanel from '../components/DebatePanel';
import PriceChart from '../components/PriceChart';
import ReactMarkdown from 'react-markdown';
import { fetchAnalysis, fetchPrice, type AnalysisData, type PriceRecord } from '../lib/api';

function extractSignal(text: string): string {
  const upper = (text || '').toUpperCase();
  for (const s of ['SHORT', 'COVER', 'OVERWEIGHT', 'UNDERWEIGHT', 'BUY', 'SELL', 'HOLD']) {
    if (upper.includes(s)) return s;
  }
  return 'UNKNOWN';
}

function extractEntryPrice(text: string): number | null {
  // Look for price patterns like $67,000, $67000, 67000, entry at $67000
  const patterns = [
    /entry\s+(?:price\s+)?[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /(?:enter|entry)\s+(?:at\s+)?[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /(?:price|level)\s*[:@\s]*\$?([\d,]+(?:\.\d+)?)/i,
    /\$?([\d,]{3,}(?:\.\d+)?)/,  // fallback: any number with 3+ digits
  ];
  
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const priceStr = match[1].replace(/,/g, '');
      const price = parseFloat(priceStr);
      if (!isNaN(price) && price > 0) {
        return price;
      }
    }
  }
  return null;
}

export default function AnalysisDetail() {
  const { ticker, date } = useParams<{ ticker: string; date: string }>();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [priceData, setPriceData] = useState<PriceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'reports' | 'debates'>('overview');

  useEffect(() => {
    if (!ticker || !date) return;
    setLoading(true);
    Promise.all([
      fetchAnalysis(ticker, date).catch(() => null),
      fetchPrice(ticker, 90).catch(() => []),
    ]).then(([a, p]) => {
      setAnalysis(a);
      setPriceData(p);
    }).finally(() => setLoading(false));
  }, [ticker, date]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-20 text-center text-slate-500">
        Loading analysis...
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-20 text-center">
        <p className="text-slate-500">Analysis not found for {ticker} on {date}</p>
        <button onClick={() => navigate('/history')} className="mt-4 text-accent-teal text-sm hover:underline">
          Back to History
        </button>
      </div>
    );
  }

  const signal = extractSignal(analysis.final_trade_decision);

  // Try to extract entry price from trader investment decision or final decision
  const priceFromText = extractEntryPrice(analysis.trader_investment_decision || '') 
    || extractEntryPrice(analysis.final_trade_decision || '');
  
  // Get closing price on analysis date as fallback
  const closePrice = priceData.find(p => p.date === date)?.close 
    || priceData[priceData.length - 1]?.close 
    || 0;
  
  const entryPrice = priceFromText || closePrice;

  const signalMarkers = [{
    date: date!,
    signal,
    price: entryPrice,
  }];

  const TABS = [
    { key: 'overview', label: 'Overview' },
    { key: 'reports', label: 'Agent Reports' },
    { key: 'debates', label: 'Debates' },
  ] as const;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate(`/history/${ticker}`)} className="text-slate-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{ticker}</h1>
          <p className="text-sm text-slate-500">{date}</p>
        </div>
        <SignalBadge signal={signal} size="lg" />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-navy-800/50 p-1 rounded-lg w-fit">
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-accent-teal/10 text-accent-teal'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          {/* Price chart */}
          {priceData.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-3">Price Chart (90d)</h2>
              <PriceChart data={priceData} signals={signalMarkers} height={350} />
            </div>
          )}

          {/* Decision card */}
          <div className="glass p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-white">Final Decision</h2>
              <SignalBadge signal={signal} size="md" />
            </div>
            <div className="markdown-content text-sm max-h-[500px] overflow-y-auto">
              <ReactMarkdown>{analysis.final_trade_decision}</ReactMarkdown>
            </div>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Market Report', value: analysis.market_report ? '✓ Available' : '—', color: 'text-accent-teal' },
              { label: 'Sentiment Report', value: analysis.sentiment_report ? '✓ Available' : '—', color: 'text-accent-cyan' },
              { label: 'News Report', value: analysis.news_report ? '✓ Available' : '—', color: 'text-accent-amber' },
              { label: 'Fundamentals', value: analysis.fundamentals_report ? '✓ Available' : '—', color: 'text-accent-purple' },
            ].map(stat => (
              <div key={stat.label} className="glass p-4">
                <p className="text-xs text-slate-500 mb-1">{stat.label}</p>
                <p className={`text-sm font-medium ${stat.color}`}>{stat.value}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {activeTab === 'reports' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          <AgentReportCard
            title="Market Analysis"
            icon={<BarChart3 className="w-4 h-4" />}
            content={analysis.market_report}
            defaultOpen={true}
          />
          <AgentReportCard
            title="Social & Sentiment"
            icon={<MessageSquare className="w-4 h-4" />}
            content={analysis.sentiment_report}
          />
          <AgentReportCard
            title="News Analysis"
            icon={<Newspaper className="w-4 h-4" />}
            content={analysis.news_report}
          />
          <AgentReportCard
            title="Fundamentals"
            icon={<PieChart className="w-4 h-4" />}
            content={analysis.fundamentals_report}
          />

          {/* Trader plan */}
          {analysis.trader_investment_decision && (
            <div className="glass p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Trader Investment Plan</h3>
              <div className="markdown-content text-sm max-h-64 overflow-y-auto">
                <ReactMarkdown>{analysis.trader_investment_decision}</ReactMarkdown>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {activeTab === 'debates' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-3">Investment Debate (Bull vs Bear)</h2>
            <DebatePanel
              type="investment"
              data={{
                bull_history: analysis.investment_debate_state?.bull_history,
                bear_history: analysis.investment_debate_state?.bear_history,
                judge_decision: analysis.investment_debate_state?.judge_decision,
              }}
            />
          </div>
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-3">Risk Debate</h2>
            <DebatePanel
              type="risk"
              data={{
                aggressive_history: analysis.risk_debate_state?.aggressive_history,
                conservative_history: analysis.risk_debate_state?.conservative_history,
                neutral_history: analysis.risk_debate_state?.neutral_history,
                judge_decision: analysis.risk_debate_state?.judge_decision,
              }}
            />
          </div>

          {/* Investment Plan (Judge) */}
          {analysis.investment_plan && (
            <div className="glass p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Investment Plan (Research Manager)</h3>
              <div className="markdown-content text-sm max-h-64 overflow-y-auto">
                <ReactMarkdown>{analysis.investment_plan}</ReactMarkdown>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
