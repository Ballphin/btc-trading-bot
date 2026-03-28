import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Clock, Zap, TrendingUp, DollarSign, Percent, Calendar, AlertCircle, ServerOff } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

const PRESETS = [
  { ticker: 'BTC-USD', name: 'Bitcoin', color: 'bg-orange-500' },
  { ticker: 'ETH-USD', name: 'Ethereum', color: 'bg-blue-500' },
  { ticker: 'NVDA', name: 'NVIDIA', color: 'bg-green-500' },
];

const FREQUENCIES = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'biweekly', label: 'Bi-weekly' },
  { value: 'monthly', label: 'Monthly' },
];

export default function Backtest() {
  const navigate = useNavigate();
  
  // Get today's date in YYYY-MM-DD format
  const today = new Date().toISOString().split('T')[0];
  // Get date 3 months ago for default start
  const threeMonthsAgo = new Date();
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
  const defaultStartDate = threeMonthsAgo.toISOString().split('T')[0];
  
  const [ticker, setTicker] = useState('BTC-USD');
  const [startDate, setStartDate] = useState(defaultStartDate);
  const [endDate, setEndDate] = useState(today);
  const [mode, setMode] = useState<'replay' | 'simulation'>('replay');
  const [initialCapital, setInitialCapital] = useState(100000);
  const [positionSize, setPositionSize] = useState(25);
  const [frequency, setFrequency] = useState('weekly');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  // Crypto settings
  const [leverage, setLeverage] = useState(1);
  const [feeRate, setFeeRate] = useState(0.0005);
  const [positionSizing, setPositionSizing] = useState('fixed');
  const [useFunding, setUseFunding] = useState(true);
  
  // Recent backtests
  const [recentBacktests, setRecentBacktests] = useState<any[]>([]);
  
  // Active backtests (running)
  const [activeBacktests, setActiveBacktests] = useState<any[]>([]);

  // Check server health on mount and fetch backtests
  useEffect(() => {
    checkServerHealth();
    fetchRecentBacktests();
    fetchActiveBacktests();
    
    // Poll for active backtests every 3 seconds
    const interval = setInterval(() => {
      if (serverStatus === 'online') {
        fetchActiveBacktests();
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [serverStatus]);

  const fetchActiveBacktests = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/backtest/active`);
      if (response.ok) {
        const data = await response.json();
        setActiveBacktests(data.active || []);
      }
    } catch {
      // Silently fail - active backtests list is not critical
    }
  };

  const checkServerHealth = async (): Promise<boolean> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setServerStatus('online');
        setError('');
        return true;
      } else {
        setServerStatus('offline');
        setError('Server is not responding properly. Please check if the backend is running on port 8000.');
        return false;
      }
    } catch {
      setServerStatus('offline');
      setError('Cannot connect to backend server. Please ensure it is running on port 8000.');
      return false;
    }
  };

  const fetchRecentBacktests = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/backtest/results?limit=5`);
      if (response.ok) {
        const data = await response.json();
        setRecentBacktests(data.results || []);
      }
    } catch {
      // Silently fail - backtests list is not critical
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Check server before submitting
    const isOnline = await checkServerHealth();
    if (!isOnline) {
      setError('Cannot start backtest: Backend server is offline.');
      return;
    }
    
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          start_date: startDate,
          end_date: endDate,
          mode,
          config: {
            initial_capital: initialCapital,
            position_size_pct: positionSize / 100,
            frequency,
            leverage,
            maker_fee: feeRate,
            taker_fee: feeRate,
            position_sizing: positionSizing,
            use_funding: useFunding,
          },
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        if (response.status === 404) {
          throw new Error('API endpoint not found. Please check server configuration.');
        } else if (response.status === 500) {
          throw new Error(data.detail || 'Server error occurred. Check server logs.');
        }
        throw new Error(data.detail || `Failed to start backtest (${response.status})`);
      }

      const data = await response.json();
      navigate(`/backtest/${data.job_id}`);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Network error: Cannot reach the server. Please ensure the backend is running.');
        setServerStatus('offline');
      } else {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      }
      setLoading(false);
    }
  };

  // Show server offline warning
  if (serverStatus === 'offline') {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Backtest Strategy</h1>
          <p className="text-slate-400">
            Test TradingAgents strategy on historical data with simulated portfolio performance.
          </p>
        </div>

        <div className="p-6 bg-red-500/10 border border-red-500 rounded-xl">
          <div className="flex items-start gap-4">
            <ServerOff className="w-8 h-8 text-red-400 flex-shrink-0" />
            <div>
              <h2 className="text-xl font-semibold text-red-400 mb-2">Backend Server Offline</h2>
              <p className="text-slate-300 mb-4">
                The backtesting API is currently unavailable. To use this feature, you need to start the backend server.
              </p>
              <div className="bg-slate-800 p-4 rounded-lg mb-4">
                <p className="text-sm text-slate-400 mb-2">Start the server with:</p>
                <code className="text-sm text-cyan-400 font-mono">
                  python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
                </code>
              </div>
              <button
                onClick={checkServerHealth}
                className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-navy-950 font-medium rounded-lg transition-colors"
              >
                Retry Connection
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Backtest Strategy</h1>
        <p className="text-slate-400">
          Test TradingAgents strategy on historical data with simulated portfolio performance.
        </p>
      </div>

      {/* Active Backtests (Running) */}
      {activeBacktests.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <span className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" />
              Running Backtests
            </h3>
            <span className="text-xs text-slate-500">{activeBacktests.length} active</span>
          </div>
          <div className="space-y-2">
            {activeBacktests.map((bt) => (
              <div
                key={bt.job_id}
                onClick={() => navigate(`/backtest/${bt.job_id}`)}
                className="flex items-center justify-between p-3 bg-slate-800/80 rounded-lg border border-cyan-500/50 cursor-pointer hover:border-cyan-400 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" />
                  <div>
                    <div className="text-sm font-medium text-white">
                      {bt.ticker} • {bt.start_date} to {bt.end_date}
                    </div>
                    <div className="text-xs text-slate-400">
                      {bt.mode} • Click to view progress
                    </div>
                  </div>
                </div>
                <div className="text-cyan-400">
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Backtests */}
      {recentBacktests.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-slate-300">Recent Backtests</h3>
            <button
              onClick={fetchRecentBacktests}
              className="text-xs text-cyan-400 hover:text-cyan-300"
            >
              Refresh
            </button>
          </div>
          <div className="space-y-2">
            {recentBacktests.map((bt) => (
              <div
                key={bt.job_id}
                onClick={() => navigate(`/backtest/${bt.job_id}`)}
                className="flex items-center justify-between p-3 bg-slate-800 rounded-lg border border-slate-700 cursor-pointer hover:border-cyan-500 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${
                    bt.total_return_pct >= 0 ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  <div>
                    <div className="text-sm font-medium text-white">
                      {bt.ticker} • {bt.start_date} to {bt.end_date}
                    </div>
                    <div className="text-xs text-slate-400">
                      {bt.mode} • {bt.total_return_pct >= 0 ? '+' : ''}{bt.total_return_pct?.toFixed(2)}% • Sharpe: {bt.sharpe_ratio?.toFixed(2)}
                    </div>
                  </div>
                </div>
                <div className="text-cyan-400">
                  <Play className="w-4 h-4" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Asset Presets */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-slate-300 mb-3">Select Asset</label>
        <div className="flex gap-4">
          {PRESETS.map((preset) => (
            <button
              key={preset.ticker}
              onClick={() => setTicker(preset.ticker)}
              className={`flex-1 p-4 rounded-xl border-2 transition-all ${
                ticker === preset.ticker
                  ? 'border-cyan-500 bg-cyan-500/10'
                  : 'border-slate-700 bg-slate-800 hover:border-slate-600'
              }`}
            >
              <div className={`w-3 h-3 rounded-full ${preset.color} mb-2`} />
              <div className="text-white font-semibold">{preset.name}</div>
              <div className="text-sm text-slate-400">{preset.ticker}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Mode Selection */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-slate-300 mb-3">Backtest Mode</label>
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={() => setMode('replay')}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              mode === 'replay'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-700 bg-slate-800 hover:border-slate-600'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              <span className="font-semibold text-white">Quick Replay</span>
            </div>
            <p className="text-sm text-slate-400">
              Uses cached analysis decisions. Results in &lt;2 seconds.
            </p>
          </button>
          <button
            onClick={() => setMode('simulation')}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              mode === 'simulation'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-700 bg-slate-800 hover:border-slate-600'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-5 h-5 text-cyan-400" />
              <span className="font-semibold text-white">Full Simulation</span>
            </div>
            <p className="text-sm text-slate-400">
              Runs full LLM pipeline on each date. Takes several minutes.
            </p>
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Date Range */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              required
            />
          </div>
        </div>

        {/* Frequency */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Trading Frequency</label>
          <div className="flex gap-2">
            {FREQUENCIES.map((freq) => (
              <button
                key={freq.value}
                type="button"
                onClick={() => setFrequency(freq.value)}
                className={`px-4 py-2 rounded-lg border transition-all ${
                  frequency === freq.value
                    ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                    : 'border-slate-700 bg-slate-800 text-slate-400 hover:border-slate-600'
                }`}
              >
                {freq.label}
              </button>
            ))}
          </div>
        </div>

        {/* Capital and Position Size */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <DollarSign className="w-4 h-4 inline mr-1" />
              Initial Capital
            </label>
            <input
              type="number"
              min="10000"
              max="500000"
              step="10000"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
            />
            <div className="mt-1 text-sm text-slate-500">
              ${initialCapital.toLocaleString()}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              <Percent className="w-4 h-4 inline mr-1" />
              Position Size
            </label>
            <input
              type="range"
              min="5"
              max="100"
              value={positionSize}
              onChange={(e) => setPositionSize(Number(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="mt-1 text-sm text-slate-500">{positionSize}% per trade</div>
          </div>
        </div>

        {/* Crypto Trading Settings */}
        <div className="border border-slate-700 rounded-xl p-4 space-y-4">
          <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Crypto Trading Settings
          </h3>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Leverage</label>
              <select
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              >
                <option value={1}>1x (No Leverage)</option>
                <option value={2}>2x</option>
                <option value={3}>3x</option>
                <option value={5}>5x</option>
                <option value={10}>10x</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Position Sizing</label>
              <select
                value={positionSizing}
                onChange={(e) => setPositionSizing(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              >
                <option value="fixed">Fixed %</option>
                <option value="kelly">Kelly Criterion</option>
                <option value="volatility">Volatility Adj</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Fee Rate</label>
              <select
                value={feeRate}
                onChange={(e) => setFeeRate(Number(e.target.value))}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              >
                <option value={0.0002}>Maker 0.02%</option>
                <option value={0.0005}>Taker 0.05%</option>
                <option value={0.001}>High 0.1%</option>
              </select>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="useFunding"
              checked={useFunding}
              onChange={(e) => setUseFunding(e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500"
            />
            <label htmlFor="useFunding" className="text-sm text-slate-400">
              Include funding rate costs (shorts pay ~0.01% per 8h)
            </label>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500 rounded-lg text-red-400 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-4 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-700 text-navy-950 font-bold rounded-xl flex items-center justify-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-navy-950/30 border-t-navy-950 rounded-full animate-spin" />
              Starting Backtest...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Start Backtest
            </>
          )}
        </button>
      </form>
    </div>
  );
}
