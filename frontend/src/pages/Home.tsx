import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Clock, ArrowRight } from 'lucide-react';
import { fetchTickers, type TickerInfo } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';
import ModelSelectorTicker from '../components/ModelSelectorTicker';

const QUICK_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'NVDA', 'AAPL', 'TSLA'];

export default function Home() {
  useDocumentTitle('Home');
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  
  // HIGH FIX: Model configuration state
  type Provider = 'openrouter' | 'deepseek' | 'openai' | 'anthropic';
  const [modelConfig, setModelConfig] = useState<{
    provider: Provider;
    model: string;
    parallelMode: boolean;
  }>({
    provider: 'openrouter',
    model: 'qwen/qwen3.6-plus',
    parallelMode: true,
  });

  useEffect(() => {
    fetchTickers().then(setTickers).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const ticker = query.trim().toUpperCase();
    if (ticker) {
      // HIGH FIX: Pass model config to analysis page
      navigate(`/analyze/${ticker}`, { state: { modelConfig } });
    }
  }, [query, navigate, modelConfig]);

  // HIGH FIX: Pass model config when navigating
  const handleQuick = (ticker: string) => {
    navigate(`/analyze/${ticker}`, { state: { modelConfig } });
  };
  
  const handleConfigChange = (newConfig: typeof modelConfig) => {
    setModelConfig(newConfig);
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex flex-col">
      {/* HIGH FIX: Model Selector Ticker */}
      <ModelSelectorTicker 
        currentTicker={query.toUpperCase() || undefined}
        onConfigChange={handleConfigChange}
      />
      
      {/* Hero */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-20">
        <div className="text-center max-w-2xl animate-fade-in-up">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-8 leading-tight">
            Run multi-agent analysis
          </h1>

          {/* Search */}
          <form onSubmit={handleSubmit} className="relative max-w-md mx-auto mb-6">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Enter ticker (e.g. BTC-USD, NVDA)"
              className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-navy-800 border border-white/10 text-white placeholder:text-slate-500 focus:outline-none focus:border-accent-teal/50 focus:ring-1 focus:ring-accent-teal/20 text-base"
              autoFocus
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 rounded-lg bg-gradient-to-r from-accent-teal to-accent-cyan text-navy-950 font-bold text-sm hover:opacity-90 transition-opacity"
            >
              Analyze
            </button>
          </form>

          {/* Quick tickers */}
          <div className="flex flex-wrap items-center justify-center gap-2">
            {QUICK_TICKERS.map(t => (
              <button
                key={t}
                onClick={() => handleQuick(t)}
                className="px-3 py-1.5 min-h-[44px] rounded-lg bg-navy-800 border border-white/5 text-slate-300 text-xs font-medium hover:border-accent-teal/30 hover:text-white transition-colors"
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Recent analyses */}
      {tickers.length > 0 && (
        <div className="px-6 pb-12 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-4 h-4 text-slate-500" />
              <h2 className="text-sm font-medium text-slate-400">Recent Analyses</h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {tickers.map((t, i) => (
                <div
                  key={t.ticker}
                  onClick={() => navigate(`/history/${t.ticker}`)}
                  className="glass-static p-4 cursor-pointer group animate-fade-in-up"
                  style={{ animationDelay: `${i * 0.05}s` }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-white text-lg">{t.ticker}</span>
                    <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-accent-teal transition-colors" />
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-500">
                      {t.analysis_count} {t.analysis_count === 1 ? 'analysis' : 'analyses'}
                    </span>
                    {t.latest_date && (
                      <span className="text-slate-500">Latest: {t.latest_date}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center pb-12 text-slate-500 text-sm">Loading recent analyses...</div>
      )}
    </div>
  );
}
