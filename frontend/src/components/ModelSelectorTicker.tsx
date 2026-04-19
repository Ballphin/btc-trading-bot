import { useState, useEffect, useCallback } from 'react';
import { Cpu, RefreshCw, DollarSign, Save, Check, Lock } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

// Providers locked to DeepSeek in production (backend enforces via /api/model/config).
// Keep the list in sync with server.py `_ALLOWED_PROVIDERS`.
const LOCKED_ALLOWED_PROVIDERS = new Set(['deepseek']);

interface ModelConfig {
  provider: 'openrouter' | 'deepseek' | 'openai' | 'anthropic';
  model: string;
  parallelMode: boolean;
}

interface ModelSelectorTickerProps {
  currentTicker?: string;
  onConfigChange?: (config: ModelConfig) => void;
}

const PROVIDER_MODELS: Record<string, string[]> = {
  openrouter: ['google/gemma-4-26b-a4b-it', 'google/gemma-4-26b-a4b-it:free', 'google/gemma-4-31b-it', 'google/gemma-4-31b-it:free', 'qwen/qwen3.6-plus', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o'],
  deepseek: ['deepseek-chat', 'deepseek-coder'],
  openai: ['gpt-5.2', 'gpt-5-mini'],
  anthropic: ['claude-3-5-sonnet-20241022'],
};

const PROVIDER_NAMES: Record<string, string> = {
  openrouter: 'OpenRouter',
  deepseek: 'DeepSeek',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
};

const ENSEMBLE_ENABLED_PROVIDERS = ['openrouter'];
const ENSEMBLE_DISABLED_PROVIDERS = ['deepseek'];

export default function ModelSelectorTicker({ currentTicker, onConfigChange }: ModelSelectorTickerProps) {
  const [config, setConfig] = useState<ModelConfig>({
    provider: 'deepseek',
    model: 'deepseek-chat',
    parallelMode: false,
  });
  const [savedConfig, setSavedConfig] = useState<ModelConfig>(config);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [lockError, setLockError] = useState<string | null>(null);

  const isDirty = config.provider !== savedConfig.provider
    || config.model !== savedConfig.model
    || config.parallelMode !== savedConfig.parallelMode;

  // Load config on mount. Drop stale localStorage that violates the lock.
  useEffect(() => {
    const saved = localStorage.getItem('modelConfig');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (LOCKED_ALLOWED_PROVIDERS.has((parsed.provider || '').toLowerCase())) {
          setConfig(parsed);
          setSavedConfig(parsed);
          onConfigChange?.(parsed);
          return;
        }
        // Stale non-deepseek state — clear it and re-fetch from backend
        localStorage.removeItem('modelConfig');
      } catch (e) {
        console.error('Failed to parse saved model config:', e);
      }
    }
    fetchModelConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchModelConfig = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/model/config`);
      if (res.ok) {
        const data = await res.json();
        const newConfig: ModelConfig = {
          provider: data.provider || 'deepseek',
          model: data.model || 'deepseek-chat',
          parallelMode: data.ensemble_enabled ?? false,
        };
        setConfig(newConfig);
        setSavedConfig(newConfig);
      }
    } catch (e) {
      console.error('Failed to fetch model config:', e);
    }
  };

  const handleSave = useCallback(async () => {
    setIsLoading(true);
    setLockError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/model/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: config.provider,
          model: config.model,
          parallel_mode: config.parallelMode,
        }),
      });
      if (res.ok) {
        setSavedConfig(config);
        localStorage.setItem('modelConfig', JSON.stringify(config));
        onConfigChange?.(config);
        setJustSaved(true);
        setTimeout(() => setJustSaved(false), 2000);
      } else {
        // Surface structured PROVIDER_LOCKED / MODEL_NOT_ALLOWED errors
        let msg = 'Failed to save model config';
        try {
          const body = await res.json();
          const detail = body?.detail;
          if (detail?.error_code === 'PROVIDER_LOCKED' || detail?.error_code === 'MODEL_NOT_ALLOWED') {
            msg = detail.message || msg;
          } else if (typeof detail === 'string') {
            msg = detail;
          }
        } catch { /* keep default */ }
        setLockError(msg);
        console.error(msg);
      }
    } catch (e) {
      console.error('Failed to save model config:', e);
      setLockError('Network error while saving model config.');
    } finally {
      setIsLoading(false);
    }
  }, [config, onConfigChange]);

  const handleProviderChange = (provider: string) => {
    const newProvider = provider as ModelConfig['provider'];
    const models = PROVIDER_MODELS[newProvider];
    const newModel = models[0];
    const canEnsemble = !ENSEMBLE_DISABLED_PROVIDERS.includes(newProvider);
    
    setConfig({
      provider: newProvider,
      model: newModel,
      parallelMode: canEnsemble && ENSEMBLE_ENABLED_PROVIDERS.includes(newProvider),
    });
  };

  const handleModelChange = (model: string) => {
    setConfig(prev => ({ ...prev, model }));
  };

  const handleParallelToggle = () => {
    if (ENSEMBLE_DISABLED_PROVIDERS.includes(config.provider)) return;
    setConfig(prev => ({ ...prev, parallelMode: !prev.parallelMode }));
  };

  const isEnsembleDisabled = ENSEMBLE_DISABLED_PROVIDERS.includes(config.provider);
  const isEnsembleSupported = ENSEMBLE_ENABLED_PROVIDERS.includes(config.provider);

  return (
    <div className="w-full bg-navy-800 border-b border-white/10">
      {/* Ticker Bar */}
      <div 
        className="flex items-center gap-4 px-4 py-2 text-xs cursor-pointer hover:bg-navy-700/50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-accent-cyan" />
          <span className="font-mono text-slate-300">
            MODEL: <span className="text-accent-cyan font-medium">{config.model}</span>
          </span>
        </div>
        
        <div className="h-3 w-px bg-white/10" />
        
        <div className="flex items-center gap-2">
          <span className="font-mono text-slate-400">
            PROVIDER: <span className="text-slate-200">{PROVIDER_NAMES[config.provider].toUpperCase()}</span>
          </span>
        </div>
        
        <div className="h-3 w-px bg-white/10" />
        
        <div className="flex items-center gap-2">
          <RefreshCw 
            className={`w-3.5 h-3.5 ${config.parallelMode ? 'text-accent-teal' : 'text-slate-500'}`} 
          />
          <span className={`font-mono ${config.parallelMode ? 'text-accent-teal' : 'text-slate-500'}`}>
            ENSEMBLE: {config.parallelMode ? '3x ON' : 'OFF'}
          </span>
        </div>
        
        {config.parallelMode && (
          <>
            <div className="h-3 w-px bg-white/10" />
            <div className="flex items-center gap-1.5">
              <DollarSign className="w-3 h-3 text-amber-400" />
              <span className="text-amber-400 font-mono">3x API CALLS</span>
            </div>
          </>
        )}
        
        {currentTicker && (
          <>
            <div className="h-3 w-px bg-white/10" />
            <span className="text-slate-500 font-mono">
              TICKER: {currentTicker}
            </span>
          </>
        )}
        
        <div className="flex-1" />
        
        <div className="flex items-center gap-1 text-slate-500 text-[10px]">
          {isDirty && <span className="text-amber-400 font-medium">● unsaved</span>}
          <span>{isExpanded ? '▼ click to close' : '▶ click to change'}</span>
        </div>
      </div>

      {/* Expanded Controls */}
      {isExpanded && (
        <div className="px-4 py-3 border-t border-white/5 bg-navy-800/50">
          <div className="flex items-start gap-4">
            {/* Provider Select */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                Provider
              </label>
              <div className="flex items-center gap-2">
                <select
                  value={config.provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                  className="bg-navy-900 border border-white/10 rounded px-3 py-1.5 text-xs text-slate-200 focus:border-accent-teal focus:outline-none"
                >
                  <option value="deepseek">DeepSeek</option>
                  <option value="openrouter" disabled>OpenRouter (locked)</option>
                  <option value="openai" disabled>OpenAI (locked)</option>
                  <option value="anthropic" disabled>Anthropic (locked)</option>
                </select>
                <Lock
                  className="w-3 h-3 text-slate-500"
                  aria-label="Provider locked for this deployment"
                />
              </div>
            </div>

            {/* Model Select */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                Model
              </label>
              <select
                value={config.model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="bg-navy-900 border border-white/10 rounded px-3 py-1.5 text-xs text-slate-200 focus:border-accent-teal focus:outline-none min-w-[180px]"
              >
                {PROVIDER_MODELS[config.provider].map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            {/* Ensemble Toggle */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-slate-500 uppercase tracking-wider">
                Ensemble (3x Parallel)
              </label>
              <button
                onClick={handleParallelToggle}
                disabled={isEnsembleDisabled}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  isEnsembleDisabled
                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    : config.parallelMode
                    ? 'bg-accent-teal/20 text-accent-teal border border-accent-teal/30'
                    : 'bg-navy-900 text-slate-400 border border-white/10 hover:border-white/20'
                }`}
              >
                {isEnsembleDisabled 
                  ? 'Not Available' 
                  : config.parallelMode ? 'ENABLED' : 'DISABLED'}
              </button>
              {isEnsembleDisabled && (
                <span className="text-[10px] text-slate-500">
                  DeepSeek does not support ensemble
                </span>
              )}
              {isEnsembleSupported && config.parallelMode && (
                <span className="text-[10px] text-amber-400">
                  Uses 3x API calls
                </span>
              )}
            </div>

            {/* Save Button */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-slate-500 uppercase tracking-wider">&nbsp;</label>
              <button
                onClick={handleSave}
                disabled={!isDirty && !justSaved}
                className={`flex items-center gap-1.5 px-4 py-1.5 rounded text-xs font-semibold transition-all ${
                  justSaved
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : isDirty
                    ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40 hover:bg-accent-cyan/30 shadow-[0_0_12px_rgba(6,214,160,0.15)]'
                    : 'bg-navy-900 text-slate-500 border border-white/5 cursor-default'
                }`}
              >
                {isLoading ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : justSaved ? (
                  <Check className="w-3.5 h-3.5" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
                {isLoading ? 'Saving...' : justSaved ? 'Saved!' : isDirty ? 'Save' : 'Saved'}
              </button>
            </div>

            {/* Info Box */}
            <div className="flex-1 ml-4 p-2 bg-navy-900/50 rounded border border-white/5">
              <p className="text-[10px] text-slate-400 leading-relaxed">
                <span className="text-accent-teal font-medium">Ensemble Mode:</span> Runs 3 parallel analyses 
                with consensus averaging. Reduces variance in results. 
                {config.parallelMode 
                  ? ' Currently active for this ticker.' 
                  : ' Disabled - single analysis will run.'}
                {isDirty && (
                  <span className="block mt-1 text-amber-400 font-medium">⚠ Unsaved changes — click Save to apply.</span>
                )}
                {lockError && (
                  <span className="block mt-1 text-red-400 font-medium">🔒 {lockError}</span>
                )}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
