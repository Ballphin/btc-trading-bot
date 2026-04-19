import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FileText } from 'lucide-react';
import useDocumentTitle from '../hooks/useDocumentTitle';
import { API_BASE_URL } from '../lib/api';

interface BacktestSummary {
  id: string;
  ticker: string;
  start_date: string;
  end_date: string;
  frequency: string;
  mode: string;
  created_at: string;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate_pct: number;
}

export default function RecentBacktests() {
  useDocumentTitle('Backtest History');
  const [backtests, setBacktests] = useState<BacktestSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRecentBacktests();
  }, []);

  const fetchRecentBacktests = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/backtests/recent`);
      if (response.ok) {
        const data = await response.json();
        setBacktests(data);
      }
    } catch (error) {
      console.error('Error fetching recent backtests:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatDateTime = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="text-center text-slate-400">Loading backtests...</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-bold text-white mb-6">Backtest History</h1>

      {backtests.length === 0 ? (
        <div className="glass p-12 text-center">
          <FileText className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400 mb-4">No backtests run yet</p>
          <Link
            to="/backtest"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-accent-teal to-accent-cyan hover:opacity-90 text-navy-950 font-bold rounded-lg transition-opacity text-sm"
          >
            Run Backtest
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {backtests.map((backtest) => (
            <Link
              key={backtest.id}
              to={`/backtest/results/${backtest.id}`}
              className="glass-static p-5 flex items-center justify-between gap-6 hover:bg-white/5 transition-colors animate-fade-in-up"
              style={{ display: 'flex' }}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <span className="text-lg font-bold text-white font-mono">{backtest.ticker}</span>
                  <span className="text-xs text-slate-500">{formatDate(backtest.start_date)} – {formatDate(backtest.end_date)}</span>
                  <span className="text-xs text-accent-teal capitalize">{backtest.frequency}</span>
                </div>
                <span className="text-xs text-slate-600">{formatDateTime(backtest.created_at)}</span>
              </div>

              <div className="flex items-center gap-8 shrink-0">
                <div className="text-right">
                  <div className="text-xs text-slate-500 mb-0.5">Sharpe</div>
                  <div className="text-sm font-semibold text-white tabular-nums">{backtest.sharpe_ratio.toFixed(2)}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-500 mb-0.5">Drawdown</div>
                  <div className="text-sm font-semibold text-red-400 tabular-nums">−{backtest.max_drawdown_pct.toFixed(2)}%</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-500 mb-0.5">Win Rate</div>
                  <div className="text-sm font-semibold text-white tabular-nums">{backtest.win_rate_pct.toFixed(1)}%</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-500 mb-0.5">Trades</div>
                  <div className="text-sm font-semibold text-white tabular-nums">{backtest.total_trades}</div>
                </div>
                <div className={`text-xl font-bold tabular-nums ${backtest.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {backtest.total_return_pct >= 0 ? '+' : ''}{backtest.total_return_pct.toFixed(2)}%
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
