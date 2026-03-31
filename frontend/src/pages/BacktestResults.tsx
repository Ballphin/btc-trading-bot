import { useEffect, useState, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown, Activity, DollarSign, Percent, BarChart3, Calendar, Download, Brain, ChevronDown, ChevronUp } from 'lucide-react';
import { API_BASE_URL, type BacktestResult, type PriceRecord, fetchPrice } from '../lib/api';
import PriceChart from '../components/PriceChart';
import EquityCurveChart from '../components/EquityCurveChart';

interface BacktestJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  result?: BacktestResult;
  error?: string;
}


interface StatusStep {
  step: number;
  total_steps: number;
  status: string;
  details?: string;
  timestamp: string;
}

export default function BacktestResults() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<BacktestJob | null>(null);
  const [priceData, setPriceData] = useState<PriceRecord[]>([]);
  const [progress, setProgress] = useState({ current: 0, total: 0, date: '' });
  const [decisions, setDecisions] = useState<any[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [statusSteps, setStatusSteps] = useState<StatusStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [lessons, setLessons] = useState<any[]>([]);
  const [lessonsExpanded, setLessonsExpanded] = useState(true);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Fetch lessons whenever job result is loaded (ticker becomes known)
  const resultTicker = job?.result?.config?.ticker;
  useEffect(() => {
    if (!resultTicker) return;
    fetch(`${API_BASE_URL}/backtests/lessons/${resultTicker}`)
      .then(r => r.json())
      .then(data => setLessons(data.lessons || []))
      .catch(() => {});
  }, [resultTicker]);

  useEffect(() => {
    if (!jobId) return;

    // First, try to fetch saved results (for completed backtests)
    const fetchSavedResult = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/backtest/${jobId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'completed' && data.result) {
            // Found saved results - no need to connect to SSE
            setJob(data);
            setLoading(false);
            setError('');
            // Fetch price data for the chart
            if (data.result?.config?.ticker) {
              const ticker = data.result.config.ticker;
              const startDate = data.result.config.start_date;
              const endDate = data.result.config.end_date;
              const days = Math.ceil((new Date(endDate).getTime() - new Date(startDate).getTime()) / (1000 * 60 * 60 * 24)) + 30;
              fetchPrice(ticker, Math.max(days, 90))
                .then(setPriceData)
                .catch(console.error);
            }
            return true; // Found saved result
          }
        }
      } catch (err) {
        console.error('Error fetching saved result:', err);
      }
      return false; // No saved result found
    };

    // Try saved results first
    fetchSavedResult().then((foundSaved) => {
      if (foundSaved) return; // Don't connect to SSE if we have saved results

      // Only connect to SSE for active/running backtests
      const eventSource = new EventSource(`${API_BASE_URL}/backtest/stream/${jobId}`);
      eventSourceRef.current = eventSource;

    const handleEvent = (eventData: any) => {
      try {
        switch (eventData.event) {
          case 'status':
            const step: StatusStep = {
              step: eventData.step,
              total_steps: eventData.total_steps,
              status: eventData.status,
              details: eventData.details,
              timestamp: new Date().toISOString(),
            };
            setStatusSteps((prev) => {
              const filtered = prev.filter((s) => s.step !== step.step);
              return [...filtered, step].sort((a, b) => a.step - b.step);
            });
            setCurrentStep(eventData.step);
            break;
          
          case 'progress':
            setProgress({
              current: eventData.current,
              total: eventData.total,
              date: eventData.date,
            });
            break;
          
          case 'decision':
            setDecisions((prev) => [...prev, eventData]);
            break;
          
          case 'complete':
            setJob({
              job_id: jobId,
              status: 'completed',
              result: eventData.result,
            });
            setLoading(false);
            eventSource.close();
            // Fetch price data for the chart
            if (eventData.result?.config?.ticker) {
              const ticker = eventData.result.config.ticker;
              const startDate = eventData.result.config.start_date;
              const endDate = eventData.result.config.end_date;
              const days = Math.ceil((new Date(endDate).getTime() - new Date(startDate).getTime()) / (1000 * 60 * 60 * 24)) + 30;
              fetchPrice(ticker, Math.max(days, 90))
                .then(setPriceData)
                .catch(console.error);
            }
            break;
          
          case 'error':
            setError(eventData.message || 'Backtest failed');
            setLoading(false);
            eventSource.close();
            break;
        }
      } catch (err) {
        console.error('Error handling SSE event:', err);
      }
    };

    // Listen to named events (sse_starlette sends named events)
    for (const eventName of ['status', 'progress', 'decision', 'complete', 'error', 'ping']) {
      eventSource.addEventListener(eventName, (msg: MessageEvent) => {
        try {
          const data = JSON.parse(msg.data);
          handleEvent(data);
        } catch (err) {
          console.error(`Error parsing SSE event '${eventName}':`, err);
        }
      });
    }

    // Fallback: onmessage for unnamed events
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEvent(data);
      } catch (err) {
        console.error('Error parsing SSE message:', err);
      }
    };

    eventSource.onerror = () => {
      // Don't show error if data was already fetched successfully
      // The SSE stream may 404 for completed jobs that are no longer in memory
      setLoading(false);
      eventSource.close();
    };

      return () => {
        eventSource.close();
      };
    });

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [jobId]);

  const handleDownload = () => {
    if (!job?.result) return;
    const blob = new Blob([JSON.stringify(job.result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backtest_${jobId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading && !job?.result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Link to="/backtest" className="flex items-center gap-2 text-slate-400 hover:text-white mb-6">
          <ArrowLeft className="w-4 h-4" />
          Back to Backtest
        </Link>

        <div className="bg-slate-800 rounded-xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-white mb-2">Running Backtest...</h2>
            <p className="text-slate-400">
              {currentStep > 0 && statusSteps.find(s => s.step === currentStep)?.status}
            </p>
          </div>

          {/* Step Progress */}
          <div className="max-w-2xl mx-auto mb-8">
            <div className="flex items-center justify-between mb-4">
              {[1, 2, 3, 4, 5].map((step, idx) => (
                <div key={step} className="flex items-center">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all duration-300 ${
                      step < currentStep
                        ? 'bg-green-500 text-white'
                        : step === currentStep
                        ? 'bg-cyan-500 text-white animate-pulse'
                        : 'bg-slate-700 text-slate-400'
                    }`}
                  >
                    {step < currentStep ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      step
                    )}
                  </div>
                  {idx < 4 && (
                    <div
                      className={`w-12 h-1 mx-1 transition-all duration-300 ${
                        step < currentStep ? 'bg-green-500' : 'bg-slate-700'
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="flex justify-between text-xs text-slate-400">
              <span className="w-16 text-center">Init</span>
              <span className="w-16 text-center">Dates</span>
              <span className="w-16 text-center">Process</span>
              <span className="w-16 text-center">Metrics</span>
              <span className="w-16 text-center">Save</span>
            </div>
          </div>

          {/* Progress Bar */}
          {progress.total > 0 && (
            <div className="max-w-md mx-auto mb-6">
              <div className="flex justify-between text-sm text-slate-400 mb-2">
                <span>Processing {progress.date}</span>
                <span>{progress.current} / {progress.total}</span>
              </div>
              <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 transition-all duration-300"
                  style={{ width: `${(progress.current / progress.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Status Log */}
          {statusSteps.length > 0 && (
            <div className="max-w-2xl mx-auto">
              <h3 className="text-sm font-medium text-slate-400 mb-3">Operation Log</h3>
              <div className="bg-slate-900 rounded-lg p-4 max-h-64 overflow-y-auto space-y-2">
                {statusSteps.map((step, idx) => (
                  <div
                    key={idx}
                    className={`flex items-start gap-3 text-sm ${
                      step.step === currentStep ? 'text-cyan-400' : 'text-slate-400'
                    }`}
                  >
                    <span className="text-xs text-slate-500 whitespace-nowrap">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                      step.step < currentStep
                        ? 'bg-green-500/20 text-green-400'
                        : step.step === currentStep
                        ? 'bg-cyan-500/20 text-cyan-400'
                        : 'bg-slate-700 text-slate-500'
                    }`}>
                      {step.step < currentStep ? '✓' : step.step}
                    </div>
                    <div>
                      <span className="font-medium">{step.status}</span>
                      {step.details && (
                        <p className="text-xs text-slate-500 mt-0.5">{step.details}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Latest Decisions */}
          {decisions.length > 0 && (
            <div className="mt-6 text-left max-w-md mx-auto">
              <p className="text-sm text-slate-400 mb-2">Latest decisions:</p>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {decisions.slice(-5).map((decision, idx) => (
                  <div
                    key={idx}
                    className={`p-2 rounded text-sm ${
                      decision.signal === 'BUY' || decision.signal === 'COVER'
                        ? 'bg-green-500/10 text-green-400'
                        : decision.signal === 'SELL' || decision.signal === 'SHORT'
                        ? 'bg-red-500/10 text-red-400'
                        : 'bg-slate-700 text-slate-300'
                    }`}
                  >
                    {decision.date}: {decision.signal} @ ${decision.price?.toLocaleString()}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (error && !job?.result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Link to="/backtest" className="flex items-center gap-2 text-slate-400 hover:text-white mb-6">
          <ArrowLeft className="w-4 h-4" />
          Back to Backtest
        </Link>
        <div className="bg-red-500/10 border border-red-500 rounded-xl p-8 text-red-400">
          <h2 className="text-xl font-semibold mb-2">Backtest Failed</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!job?.result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center text-slate-400">Loading results...</div>
      </div>
    );
  }

  const { config, metrics, decisions: resultDecisions, equity_curve, trade_history } = job.result;
  const ticker = config?.ticker || 'UNKNOWN';

  // Prepare signal markers for price chart
  const signalMarkers = resultDecisions?.map((d: any) => ({
    date: d.date,
    signal: d.signal,
    price: d.price,
  })) || [];

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <Link to="/backtest" className="flex items-center gap-2 text-slate-400 hover:text-white">
          <ArrowLeft className="w-4 h-4" />
          Back to Backtest
        </Link>
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors"
        >
          <Download className="w-4 h-4" />
          Export JSON
        </button>
      </div>

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          Backtest Results: {ticker}
        </h1>
        <p className="text-slate-400">
          <Calendar className="w-4 h-4 inline mr-1" />
          {config?.start_date} to {config?.end_date} • {config?.mode} mode • {config?.frequency}
        </p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <TrendingUp className="w-4 h-4" />
            Total Return
          </div>
          <div className={`text-2xl font-bold ${metrics?.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {metrics?.total_return_pct >= 0 ? '+' : ''}{metrics?.total_return_pct?.toFixed(2)}%
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <Activity className="w-4 h-4" />
            Sharpe Ratio
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics?.sharpe_ratio?.toFixed(2) || 'N/A'}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <TrendingDown className="w-4 h-4" />
            Max Drawdown
          </div>
          <div className="text-2xl font-bold text-red-400">
            -{metrics?.max_drawdown_pct?.toFixed(2)}%
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
            <Percent className="w-4 h-4" />
            Win Rate
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics?.win_rate_pct?.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Backtest Lessons */}
      {lessons.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 mb-8 border border-cyan-500/20">
          <button
            onClick={() => setLessonsExpanded(prev => !prev)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-white">Agent Lessons Learned</h3>
              <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded-full">
                {lessons.length} insight{lessons.length !== 1 ? 's' : ''}
              </span>
            </div>
            {lessonsExpanded ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </button>
          {lessonsExpanded && (
            <div className="mt-4 space-y-3">
              <p className="text-sm text-slate-400 mb-3">
                These insights are derived from historical backtest performance and are fed into future LLM decisions.
              </p>
              {lessons.map((lesson: any, i: number) => {
                const category = (lesson.category || 'general').replace(/_/g, ' ');
                const confidence = lesson.confidence || 'medium';
                const regime = lesson.regime ? lesson.regime.replace(/_/g, ' ') : null;
                const categoryColors: Record<string, string> = {
                  'signal accuracy': 'text-green-400 bg-green-500/10',
                  'risk management': 'text-yellow-400 bg-yellow-500/10',
                  'position sizing': 'text-purple-400 bg-purple-500/10',
                };
                const colorClass = categoryColors[category] || 'text-slate-400 bg-slate-700';
                const confColor = confidence === 'high' ? 'text-green-400' : 'text-yellow-400';
                return (
                  <div key={i} className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${colorClass}`}>
                        {category}
                      </span>
                      <span className={`text-xs font-medium ${confColor}`}>
                        {confidence.toUpperCase()} CONFIDENCE
                      </span>
                      {regime && (
                        <span className="text-xs text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full capitalize">
                          {regime}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-200 leading-relaxed">{lesson.lesson}</p>
                    {lesson.win_rate != null && (
                      <div className="mt-2 flex gap-4 text-xs text-slate-400">
                        <span>Win rate: <span className="text-white font-medium">{(lesson.win_rate * 100).toFixed(1)}%</span></span>
                        {lesson.avg_return != null && (
                          <span>Avg return: <span className={lesson.avg_return >= 0 ? 'text-green-400 font-medium' : 'text-red-400 font-medium'}>{(lesson.avg_return * 100).toFixed(2)}%</span></span>
                        )}
                        {lesson.sample_size && (
                          <span>Sample: <span className="text-white font-medium">{lesson.sample_size} trades</span></span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Additional Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">Total Trades</div>
          <div className="text-xl font-semibold text-white">{metrics?.total_trades}</div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">Profit Factor</div>
          <div className="text-xl font-semibold text-white">{metrics?.profit_factor?.toFixed(2)}</div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">Initial Capital</div>
          <div className="text-xl font-semibold text-white">${config?.initial_capital?.toLocaleString()}</div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">Final Value</div>
          <div className="text-xl font-semibold text-white">${metrics?.final_value?.toLocaleString()}</div>
        </div>
      </div>

      {/* Crypto Metrics */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-white mb-4">Crypto Trading Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Leverage Used</div>
            <div className="text-xl font-semibold text-cyan-400">{config?.leverage || 1}x</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Total Fees</div>
            <div className="text-xl font-semibold text-yellow-400">${metrics?.total_fees?.toFixed(2)}</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Funding Costs</div>
            <div className={`text-xl font-semibold ${(metrics?.total_funding || 0) > 0 ? 'text-red-400' : 'text-green-400'}`}>
              ${Math.abs(metrics?.total_funding || 0).toFixed(2)}
            </div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Liquidations</div>
            <div className={`text-xl font-semibold ${(metrics?.liquidations || 0) > 0 ? 'text-red-400' : 'text-green-400'}`}>
              {metrics?.liquidations || 0}
            </div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Fee Impact</div>
            <div className="text-xl font-semibold text-white">{metrics?.fee_impact_pct?.toFixed(2)}%</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Sortino Ratio</div>
            <div className="text-xl font-semibold text-white">{metrics?.sortino_ratio?.toFixed(2)}</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Omega Ratio</div>
            <div className="text-xl font-semibold text-white">{metrics?.omega_ratio?.toFixed(2)}</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <div className="text-slate-400 text-sm mb-1">Omega Ratio</div>
            <div className="text-xl font-semibold text-white">{metrics?.omega_ratio?.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Risk Management Metrics */}
      {(metrics?.stops_hit !== undefined || metrics?.takes_hit !== undefined) && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Management</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="text-slate-400 text-sm mb-1">Stops Hit</div>
              <div className="text-xl font-semibold text-red-400">{metrics?.stops_hit || 0}</div>
            </div>
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="text-slate-400 text-sm mb-1">Targets Hit</div>
              <div className="text-xl font-semibold text-green-400">{metrics?.takes_hit || 0}</div>
            </div>
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="text-slate-400 text-sm mb-1">Avg Hold Days</div>
              <div className="text-xl font-semibold text-white">{metrics?.avg_hold_days?.toFixed(1) || 'N/A'}</div>
            </div>
            <div className="bg-slate-800 rounded-xl p-4">
              <div className="text-slate-400 text-sm mb-1">Avg R:R Ratio</div>
              <div className="text-xl font-semibold text-cyan-400">{metrics?.avg_rr_ratio?.toFixed(2) || 'N/A'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Price & Signals
          </h3>
          <PriceChart
            data={priceData}
            signals={signalMarkers}
            height={350}
          />
        </div>

        <div className="bg-slate-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-green-400" />
            Equity Curve
          </h3>
          <EquityCurveChart equityCurve={equity_curve || []} />
        </div>
      </div>

      {/* Trade Log */}
      <div className="bg-slate-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-700">
          <h3 className="text-lg font-semibold text-white">Trade Log</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Date</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Signal</th>
                <th className="px-4 py-3 text-right text-sm font-medium text-slate-300">Price</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Action</th>
                <th className="px-4 py-3 text-right text-sm font-medium text-slate-300">Portfolio Value</th>
                <th className="px-4 py-3 text-right text-sm font-medium text-slate-300">Open PnL</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-slate-300">Position</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {trade_history?.map((trade: any, idx: number) => (
                <tr key={idx} className="hover:bg-slate-700/30">
                  <td className="px-4 py-3 text-sm text-slate-300">{trade.date}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-medium rounded ${
                        trade.signal === 'BUY' || trade.signal === 'COVER'
                          ? 'bg-green-500/20 text-green-400'
                          : trade.signal === 'SELL' || trade.signal === 'SHORT'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-slate-600 text-slate-300'
                      }`}
                    >
                      {trade.signal}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 text-right">
                    ${trade.price?.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-400 truncate max-w-xs">
                    {trade.action}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 text-right">
                    ${trade.portfolio_value?.toLocaleString()}
                  </td>
                  <td className={`px-4 py-3 text-sm text-right font-medium ${trade.unrealized_pnl > 0 ? 'text-green-400' : trade.unrealized_pnl < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                    ${trade.unrealized_pnl?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || '0.00'}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-400">{trade.position}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
