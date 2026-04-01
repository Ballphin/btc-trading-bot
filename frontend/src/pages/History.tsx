import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ChevronRight, Activity } from 'lucide-react';
import SignalBadge from '../components/SignalBadge';
import { fetchTickers, fetchAnalyses, type TickerInfo, type AnalysisSummary } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';

export default function History() {
  const { ticker: selectedTicker } = useParams<{ ticker?: string }>();
  useDocumentTitle(selectedTicker ? `${selectedTicker} History` : 'Analysis History');
  const navigate = useNavigate();
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTickers().then(setTickers).catch(() => {}).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedTicker) {
      setLoading(true);
      fetchAnalyses(selectedTicker).then(setAnalyses).catch(() => setAnalyses([])).finally(() => setLoading(false));
    }
  }, [selectedTicker]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-bold text-white mb-6">Analysis History</h1>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar: Ticker list */}
        <div className="lg:col-span-1">
          <div className="glass p-2 space-y-0.5">
            {tickers.length === 0 && !loading && (
              <p className="text-slate-500 text-sm p-3">No analyses yet. Run one from the home page.</p>
            )}
            {tickers.map(t => (
              <button
                key={t.ticker}
                onClick={() => navigate(`/history/${t.ticker}`)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-left transition-colors ${
                  selectedTicker === t.ticker
                    ? 'bg-accent-teal/10 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 shrink-0" />
                  <span className="font-medium text-sm">{t.ticker}</span>
                </div>
                <span className="text-xs bg-navy-700 px-2 py-0.5 rounded-full">{t.analysis_count}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main: Analysis timeline */}
        <div className="lg:col-span-3">
          {!selectedTicker ? (
            <div className="glass p-12 text-center">
              <Activity className="w-10 h-10 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500">Select a ticker from the sidebar to view past analyses</p>
            </div>
          ) : loading ? (
            <div className="glass p-12 text-center text-slate-500">Loading analyses...</div>
          ) : analyses.length === 0 ? (
            <div className="glass p-12 text-center">
              <p className="text-slate-500">No analyses found for {selectedTicker}</p>
            </div>
          ) : (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-white mb-4">{selectedTicker} — {analyses.length} {analyses.length === 1 ? 'analysis' : 'analyses'}</h2>
              {analyses.map((a, i) => (
                <Link
                  key={a.date}
                  to={`/history/${selectedTicker}/${a.date}`}
                  className="glass-static px-4 py-3 flex items-center justify-between group no-underline animate-fade-in-up"
                  style={{ display: 'flex', animationDelay: `${i * 0.03}s` }}
                >
                  <span className="font-medium text-white text-sm tabular-nums">{a.date}</span>
                  <div className="flex items-center gap-3">
                    <SignalBadge signal={a.signal} size="sm" />
                    <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-accent-teal transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
