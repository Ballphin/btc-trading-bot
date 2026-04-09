import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, TrendingUp, AlertCircle, ServerOff } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';
import useDocumentTitle from '../hooks/useDocumentTitle';

const FREQUENCIES = [
  { value: '4h', label: '4-Hour' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'biweekly', label: 'Bi-weekly' },
  { value: 'monthly', label: 'Monthly' },
];

export default function Backtest() {
  useDocumentTitle('Backtest Strategy');
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
  
  // Active backtests (running)
  const [activeBacktests, setActiveBacktests] = useState<any[]>([]);

  const isCrypto = ticker.includes('-USD') || ticker.includes('-USDT') || ticker.includes('-BTC');

  // Check server health on mount and fetch backtests
  useEffect(() => {
    checkServerHealth();
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
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Backtest Strategy</h1>
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
                <code className="text-sm text-accent-teal font-mono">
                  python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
                </code>
              </div>
              <button
                onClick={checkServerHealth}
                className="px-4 py-2 bg-gradient-to-r from-accent-teal to-accent-cyan hover:opacity-90 text-navy-950 font-medium rounded-lg transition-colors"
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
    <div className="max-w-4xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Backtest Strategy</h1>
      </div>

      {/* Active Backtests (Running) */}
      {activeBacktests.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <span className="w-2 h-2 bg-accent-teal rounded-full animate-pulse" />
              Running Backtests
            </h3>
            <span className="text-xs text-slate-500">{activeBacktests.length} active</span>
          </div>
          <div className="space-y-2">
            {activeBacktests.map((bt) => (
              <div
                key={bt.job_id}
                onClick={() => navigate(`/backtest/${bt.job_id}`)}
                className="flex items-center justify-between p-3 bg-slate-800/80 rounded-lg border border-accent-teal/50 cursor-pointer hover:border-accent-teal transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-accent-teal rounded-full animate-pulse" />
                  <div>
                    <div className="text-sm font-medium text-white">
                      {bt.ticker} • {bt.start_date} to {bt.end_date}
                    </div>
                    <div className="text-xs text-slate-400">
                      {bt.mode} • Click to view progress
                    </div>
                  </div>
                </div>
                <div className="text-accent-teal">
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

      {/* Mode Selection */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-slate-300 mb-2">Backtest Mode</label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setMode('replay')}
            className={`flex-1 px-4 py-2.5 min-h-[44px] rounded-lg border text-sm font-medium transition-all ${
              mode === 'replay'
                ? 'border-accent-teal bg-accent-teal/10 text-accent-teal'
                : 'border-slate-700 bg-slate-800 text-slate-400 hover:border-slate-600'
            }`}
          >
            Quick Replay
            <span className="block text-xs font-normal opacity-60 mt-0.5">cached · &lt;2s</span>
          </button>
          <button
            type="button"
            onClick={() => setMode('simulation')}
            className={`flex-1 px-4 py-2.5 min-h-[44px] rounded-lg border text-sm font-medium transition-all ${
              mode === 'simulation'
                ? 'border-accent-teal bg-accent-teal/10 text-accent-teal'
                : 'border-slate-700 bg-slate-800 text-slate-400 hover:border-slate-600'
            }`}
          >
            Full Simulation
            <span className="block text-xs font-normal opacity-60 mt-0.5">full LLM · minutes</span>
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Ticker */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Ticker</label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="e.g. BTC-USD, NVDA, ETH-USD"
            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder:text-slate-500 focus:border-accent-teal/50 focus:outline-none font-mono"
            required
          />
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-accent-teal/50 focus:outline-none"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-accent-teal/50 focus:outline-none"
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
                className={`px-4 py-2 min-h-[44px] rounded-lg border transition-all ${
                  frequency === freq.value
                    ? 'border-accent-teal bg-accent-teal/10 text-accent-teal'
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
            <label className="block text-sm font-medium text-slate-300 mb-2">Initial Capital</label>
            <input
              type="number"
              min="10000"
              max="500000"
              step="10000"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-accent-teal/50 focus:outline-none"
            />
            <div className="mt-1 text-sm text-slate-500">
              ${initialCapital.toLocaleString()}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Position Size</label>
            <input
              type="range"
              min="5"
              max="100"
              value={positionSize}
              onChange={(e) => setPositionSize(Number(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-accent-teal"
            />
            <div className="mt-1 text-sm text-slate-500">{positionSize}% per trade</div>
          </div>
        </div>

        {/* Crypto Trading Settings */}
        {isCrypto && <div className="border border-slate-700 rounded-xl p-4 space-y-4">
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
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-accent-teal/50 focus:outline-none"
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
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-accent-teal/50 focus:outline-none"
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
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-accent-teal/50 focus:outline-none"
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
              className="w-4 h-4 rounded border-slate-600 bg-accent-teal/20 text-accent-teal focus:ring-accent-teal/20"
            />
            <label htmlFor="useFunding" className="text-sm text-slate-400">
              Include funding rate costs (shorts pay ~0.01% per 8h)
            </label>
          </div>
        </div>}

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
          className="w-full py-4 bg-gradient-to-r from-accent-teal to-accent-cyan hover:opacity-90 disabled:bg-slate-700 text-navy-950 font-bold rounded-xl flex items-center justify-center gap-2 transition-colors"
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
