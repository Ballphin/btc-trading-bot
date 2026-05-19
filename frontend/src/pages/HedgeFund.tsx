import { useState, useEffect, useRef, useCallback } from 'react';
import { Briefcase, AlertCircle, Play, CheckCircle2, Circle, RefreshCw, ServerOff, Timer } from 'lucide-react';
import { clsx } from 'clsx';
import { getHedgeFundAgents, startHedgeFundAnalysis, API_BASE_URL } from '../lib/api';
import type { HedgeFundAgent, HedgeFundRequest, HedgeFundResult } from '../lib/api';

function renderReasoning(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function normalizeHedgeFundError(err: unknown): string {
  const msg = typeof err === 'string'
    ? err
    : err instanceof Error
      ? err.message
      : String(err ?? 'Analysis failed');

  const lower = msg.toLowerCase();
  if (lower.includes('429') || lower.includes('too many requests')) {
    return 'NVIDIA rate limit reached (429). Please wait 30–60 seconds and retry, or select fewer analysts.';
  }

  return msg;
}

function isRateLimitError(err: unknown): boolean {
  const msg = typeof err === 'string'
    ? err
    : err instanceof Error
      ? err.message
      : String(err ?? '');
  const lower = msg.toLowerCase();
  return lower.includes('429') || lower.includes('too many requests');
}

function isLikelyCryptoTicker(ticker: string): boolean {
  const t = ticker.toUpperCase();
  return (
    t.endsWith('-USD') ||
    t.endsWith('USDT') ||
    t.endsWith('USDC') ||
    t.endsWith('-PERP') ||
    t.includes('/')
  );
}

export default function HedgeFund() {
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'waking'>('waking');
  const [agents, setAgents] = useState<HedgeFundAgent[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());
  const [tickerInput, setTickerInput] = useState('AAPL');
  const [useNvidia, setUseNvidia] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimitCooldownSec, setRateLimitCooldownSec] = useState(0);
  
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ node: string }[]>([]);
  const [result, setResult] = useState<HedgeFundResult | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  
  const eventSourceRef = useRef<EventSource | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleUiError = useCallback((err: unknown) => {
    setError(normalizeHedgeFundError(err));
    if (isRateLimitError(err)) {
      setRateLimitCooldownSec((prev) => Math.max(prev, 45));
    }
  }, []);

  useEffect(() => {
    if (rateLimitCooldownSec <= 0) return;
    const id = setInterval(() => {
      setRateLimitCooldownSec((s) => (s > 0 ? s - 1 : 0));
    }, 1000);
    return () => clearInterval(id);
  }, [rateLimitCooldownSec]);

  const checkServerHealth = useCallback(async (retryOnFail = true): Promise<boolean> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const response = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (response.ok) {
        setServerStatus('online');
        return true;
      }
      setServerStatus('offline');
      return false;
    } catch {
      if (retryOnFail) {
        setServerStatus('waking');
        // Retry once after a short delay (Render cold start)
        await new Promise(r => setTimeout(r, 5000));
        return checkServerHealth(false);
      }
      setServerStatus('offline');
      return false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    checkServerHealth().then(online => {
      if (cancelled || !online) return;
      getHedgeFundAgents()
        .then(data => {
          if (cancelled) return;
          setAgents(data);
          const initialSelected = new Set(data.slice(0, 13).map(a => a.key));
          if (initialSelected.size === 0 && data.length > 0) {
              initialSelected.add(data[0].key);
          }
          setSelectedAgents(initialSelected);
        })
        .catch(err => console.error('Failed to load agents:', err));
    });
    return () => { cancelled = true; };
  }, [checkServerHealth]);

  const toggleAgent = (key: string) => {
    const next = new Set(selectedAgents);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setSelectedAgents(next);
  };

  const cleanupStream = () => {
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
    if (elapsedRef.current) { clearInterval(elapsedRef.current); elapsedRef.current = null; }
    if (eventSourceRef.current) { eventSourceRef.current.close(); eventSourceRef.current = null; }
  };

  const handleRun = async () => {
    if (!tickerInput) {
      setError('Please enter a ticker');
      return;
    }
    const tickers = tickerInput.split(',').map(t => t.trim()).filter(Boolean);
    if (tickers.length === 0) {
      setError('Please enter at least one ticker');
      return;
    }
    const unsupported = tickers.filter(isLikelyCryptoTicker);
    if (unsupported.length > 0) {
      setError(
        `HedgeFund supports stock tickers only. Unsupported crypto tickers: ${unsupported.join(', ')}. Use the main analysis flow for crypto.`
      );
      return;
    }
    if (rateLimitCooldownSec > 0) {
      setError(`NVIDIA rate limit cooldown active. Retry in ${rateLimitCooldownSec}s.`);
      return;
    }
    if (selectedAgents.size === 0) {
      setError('Please select at least one analyst');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setProgress([]);
      setResult(null);
      setElapsedSec(0);

      const req: HedgeFundRequest = {
        tickers,
        selected_analysts: Array.from(selectedAgents),
        model_name: 'deepseek-v4-pro',
        model_provider: 'DeepSeek',
        use_nvidia_deepseek: useNvidia,
      };

      const res = await startHedgeFundAnalysis(req);
      setJobId(res.job_id);
    } catch (err: any) {
      handleUiError(err);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    cleanupStream();

    const source = new EventSource(`${API_BASE_URL}/hedgefund/stream/${jobId}`);
    eventSourceRef.current = source;

    // Elapsed timer
    const startTime = Date.now();
    elapsedRef.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    // Scaled timeout: 30s base + 30s per analyst, capped at 600s
    const timeoutMs = Math.min(30_000 + selectedAgents.size * 30_000, 600_000);
    timeoutRef.current = setTimeout(() => {
      setError('Analysis timed out — the LLM provider may be slow. Try again or select fewer analysts.');
      setLoading(false);
      cleanupStream();
    }, timeoutMs);

    source.addEventListener('progress', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        setProgress(p => [...p, data]);
      } catch {}
    });

    source.addEventListener('done', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        setResult(data);
      } catch {}
      setLoading(false);
      cleanupStream();
    });

    source.addEventListener('job_error', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        handleUiError(data.error || 'Analysis failed');
      } catch {
        setError('Analysis failed');
      }
      setLoading(false);
      cleanupStream();
    });

    // Single error handler — distinguishes native vs custom SSE errors
    source.addEventListener('error', (e: Event) => {
      const me = e as MessageEvent;
      if (me.data) {
        // Custom error event from backend
        try {
          const data = JSON.parse(me.data);
          handleUiError(data.error || 'Analysis failed');
        } catch {
          handleUiError(String(me.data));
        }
      } else {
        // Native connectivity error
        setError('Lost connection to server');
      }
      setLoading(false);
      cleanupStream();
    });

    return () => {
      cleanupStream();
    };
  }, [jobId]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center gap-3 mb-8">
        <div className="p-2.5 bg-accent-blue/10 rounded-xl">
          <Briefcase className="w-6 h-6 text-accent-blue" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">HedgeFund Analysis</h1>
          <p className="text-slate-400">Multi-agent LangGraph analysis powered by virattt/ai-hedge-fund</p>
        </div>
      </div>

      {/* Server status banners */}
      {serverStatus === 'waking' && (
        <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center gap-3">
          <div className="w-4 h-4 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
          <p className="text-sm text-amber-300">Waking up server... This may take up to 30 seconds on free tier.</p>
        </div>
      )}
      {serverStatus === 'offline' && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ServerOff className="w-5 h-5 text-red-400" />
            <div>
              <p className="text-sm text-red-300 font-medium">Server is offline</p>
              <p className="text-xs text-red-400/70 mt-0.5">Start the backend: <code className="bg-red-500/10 px-1.5 py-0.5 rounded">uvicorn server:app --port 8000</code></p>
            </div>
          </div>
          <button
            onClick={() => { setServerStatus('waking'); checkServerHealth(); }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 text-red-300 text-sm hover:bg-red-500/30 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Config Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
            <h2 className="text-lg font-medium text-white mb-4">Configuration</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Target Ticker</label>
                <input
                  type="text"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  placeholder="e.g. AAPL, MSFT"
                  className="w-full bg-slate-950 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-accent-blue"
                  disabled={loading}
                />
                <p className="text-xs text-slate-500 mt-1">Note: Works for stocks only (financialdatasets.ai)</p>
              </div>

              <div className="pt-1">
                <label className="flex items-start gap-3 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={useNvidia}
                    onChange={(e) => setUseNvidia(e.target.checked)}
                    disabled={loading}
                    className="mt-1 accent-accent-blue"
                  />
                  <span className="text-sm text-slate-300">
                    Use NVIDIA DeepSeek route
                    <span className="block text-xs text-slate-500 mt-0.5">
                      Free tier; throttled (~6s between calls) so 13 analysts run sequentially. Slower but avoids 429 errors. Default: direct DeepSeek API.
                    </span>
                  </span>
                </label>
              </div>

              <div className="pt-2">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-slate-300">Select Analysts</label>
                  <span className="text-xs text-slate-500">{selectedAgents.size} selected</span>
                </div>
                <div className="h-96 overflow-y-auto pr-2 space-y-2 custom-scrollbar">
                  {agents.map((agent) => {
                    const isSelected = selectedAgents.has(agent.key);
                    return (
                      <button
                        key={agent.key}
                        onClick={() => toggleAgent(agent.key)}
                        disabled={loading}
                        className={clsx(
                          "w-full text-left p-3 rounded-xl border transition-all",
                          isSelected 
                            ? "bg-accent-blue/10 border-accent-blue/30 text-white" 
                            : "bg-slate-950 border-white/5 text-slate-400 hover:bg-slate-900"
                        )}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-sm">{agent.display_name}</span>
                          {isSelected ? <CheckCircle2 className="w-4 h-4 text-accent-blue" /> : <Circle className="w-4 h-4 opacity-30" />}
                        </div>
                        <p className="text-xs opacity-70 line-clamp-1">{agent.description}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex gap-2">
                  <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}

              <button
                onClick={handleRun}
                disabled={loading || selectedAgents.size === 0 || !tickerInput || rateLimitCooldownSec > 0}
                className="w-full py-3 rounded-xl bg-accent-blue text-white font-medium hover:bg-accent-blue/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    {rateLimitCooldownSec > 0 ? `Retry in ${rateLimitCooldownSec}s` : 'Run HedgeFund'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Progress Stream */}
          {(loading || progress.length > 0) && !result && (
            <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
              <h3 className="text-white font-medium mb-4 flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent-blue animate-pulse" />
                  Live Execution Progress
                </span>
                {loading && elapsedSec > 0 && (
                  <span className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
                    <Timer className="w-3 h-3" />
                    {Math.floor(elapsedSec / 60)}:{String(elapsedSec % 60).padStart(2, '0')}
                  </span>
                )}
              </h3>
              <div className="space-y-3">
                {progress.map((p, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="text-slate-300">
                      <span className="text-accent-blue/80 font-mono">[{p.node}]</span> completed analysis
                    </span>
                  </div>
                ))}
                {loading && (
                  <div className="flex items-center gap-3 text-sm text-slate-500">
                    <div className="w-4 h-4 border-2 border-slate-700 border-t-slate-500 rounded-full animate-spin" />
                    Waiting for next agent...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results Display */}
          {result && (
            <div className="space-y-6">
              {result.persisted === false && (
                <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-amber-300">Result not saved to history</p>
                    <p className="text-xs text-amber-400/80 mt-1 break-words">
                      The decision below is valid but did not persist to the History tab.
                      {result.persist_error && (
                        <>
                          {' '}Reason: <code className="font-mono">{result.persist_error}</code>
                        </>
                      )}
                    </p>
                  </div>
                </div>
              )}
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <h3 className="text-xl font-bold text-white mb-2">Portfolio Manager Decision</h3>
                {Object.entries(result.decisions || {}).map(([ticker, dec]: [string, any]) => (
                  <div key={ticker} className="mt-4 p-4 rounded-xl bg-slate-950 border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="text-2xl font-bold text-white">{ticker}</div>
                      <div className={clsx(
                        "px-4 py-1.5 rounded-lg font-bold text-sm",
                        dec.action === 'BUY' ? "bg-green-500/10 text-green-400 border border-green-500/20" :
                        dec.action === 'SELL' ? "bg-red-500/10 text-red-400 border border-red-500/20" :
                        "bg-slate-500/10 text-slate-400 border border-slate-500/20"
                      )}>
                        {dec.action}
                      </div>
                    </div>
                    {dec.quantity !== undefined && (
                      <div className="text-slate-400 mb-2">Quantity: <span className="text-white">{dec.quantity}</span></div>
                    )}
                    {dec.reasoning && (
                      <div className="text-sm text-slate-300 mt-4 leading-relaxed whitespace-pre-wrap break-words">
                        {renderReasoning(dec.reasoning)}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-4">Analyst Signals</h3>
                <div className="space-y-4">
                  {Object.entries((result.analyst_signals) || {}).map(([analystName, data]: [string, any]) => {
                    const signalInfo = data[tickerInput] || data[Object.keys(data)[0]]; // fallback if ticker format differs
                    if (!signalInfo) return null;
                    return (
                      <div key={analystName} className="p-4 rounded-xl bg-slate-950 border border-white/5">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <div className="font-medium text-white">{analystName.replace('_agent', '').split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</div>
                          </div>
                          <div className={clsx(
                            "px-2 py-0.5 rounded text-xs font-medium",
                            signalInfo.signal === 'bullish' ? "bg-green-500/20 text-green-400" :
                            signalInfo.signal === 'bearish' ? "bg-red-500/20 text-red-400" :
                            "bg-slate-500/20 text-slate-400"
                          )}>
                            {signalInfo.signal?.toUpperCase() || 'NEUTRAL'}
                          </div>
                        </div>
                        {signalInfo.confidence !== undefined && (
                          <div className="text-xs text-slate-500 mb-2">Confidence: {Math.round(signalInfo.confidence)}%</div>
                        )}
                        <p className="text-sm text-slate-400 mt-2 line-clamp-3 hover:line-clamp-none transition-all whitespace-pre-wrap break-words">
                          {renderReasoning(signalInfo.reasoning)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
