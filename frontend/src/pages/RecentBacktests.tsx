import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Calendar, TrendingUp, Activity, Clock, FileText } from 'lucide-react';

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
  const [backtests, setBacktests] = useState<BacktestSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRecentBacktests();
  }, []);

  const fetchRecentBacktests = async () => {
    try {
      const response = await fetch('/api/backtests/recent');
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
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center text-slate-400">Loading backtests...</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Backtest History</h1>
        <p className="text-slate-400">View and compare your backtest results</p>
      </div>

      {backtests.length === 0 ? (
        <div className="bg-slate-800 rounded-xl p-12 text-center">
          <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-300 mb-2">No backtests yet</h3>
          <p className="text-slate-400 mb-6">Run your first backtest to see results here</p>
          <Link
            to="/backtest"
            className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors"
          >
            Run Backtest
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {backtests.map((backtest) => (
            <Link
              key={backtest.id}
              to={`/backtest/results/${backtest.id}`}
              className="bg-slate-800 rounded-xl p-6 hover:bg-slate-750 transition-colors border border-slate-700 hover:border-slate-600"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-2">{backtest.ticker}</h3>
                  <div className="flex items-center gap-1 text-base text-slate-300 font-medium mb-1">
                    <Calendar className="w-5 h-5" />
                    <span>{formatDate(backtest.start_date)} - {formatDate(backtest.end_date)}</span>
                    <span className="mx-2 text-slate-600">•</span>
                    <span className="text-cyan-400 capitalize">{backtest.frequency}</span>
                  </div>
                  <div className="flex items-center gap-1 text-sm text-slate-400">
                    <Clock className="w-4 h-4" />
                    <span>Run on {formatDateTime(backtest.created_at)}</span>
                  </div>
                </div>
                <div
                  className={`text-3xl font-bold ${
                    backtest.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {backtest.total_return_pct >= 0 ? '+' : ''}
                  {backtest.total_return_pct.toFixed(2)}%
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-slate-900 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                    <Activity className="w-3 h-3" />
                    Sharpe Ratio
                  </div>
                  <div className="text-lg font-semibold text-white">
                    {backtest.sharpe_ratio.toFixed(2)}
                  </div>
                </div>

                <div className="bg-slate-900 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                    <TrendingUp className="w-3 h-3" />
                    Max Drawdown
                  </div>
                  <div className="text-lg font-semibold text-red-400">
                    -{backtest.max_drawdown_pct.toFixed(2)}%
                  </div>
                </div>

                <div className="bg-slate-900 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Total Trades</div>
                  <div className="text-lg font-semibold text-white">{backtest.total_trades}</div>
                </div>

                <div className="bg-slate-900 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Win Rate</div>
                  <div className="text-lg font-semibold text-white">
                    {backtest.win_rate_pct.toFixed(1)}%
                  </div>
                </div>

                <div className="bg-slate-900 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Status</div>
                  <div className="text-sm font-semibold text-green-400">Completed</div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
