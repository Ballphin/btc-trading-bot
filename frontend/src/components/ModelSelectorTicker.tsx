import { useState, useEffect } from 'react';
import { Cpu, RefreshCw, DollarSign } from 'lucide-react';

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
  openrouter: ['qwen/qwen3.6-plus', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o'],
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
    provider: 'openrouter',
    model: 'qwen/qwen3.6-plus',
    parallelMode: true,
  });
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // HIGH FIX: Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('modelConfig');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setConfig(parsed);
        onConfigChange?.(parsed);
      } catch (e) {
        console.error('Failed to parse saved model config:', e);
      }
    } else {
      // Fetch from API if no localStorage
      fetchModelConfig();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // HIGH FIX: Persist to localStorage on change
  useEffect(() => {
    localStorage.setItem('modelConfig', JSON.stringify(config));
    onConfigChange?.(config);
  }, [config, onConfigChange]);

  // Auto-disable parallel for certain providers
  useEffect(() => {
    if (ENSEMBLE_DISABLED_PROVIDERS.includes(config.provider)) {
      setConfig(c => ({ ...c, parallelMode: false }));
    } else if (ENSEMBLE_ENABLED_PROVIDERS.includes(config.provider)) {
      // Auto-enable for OpenRouter if not explicitly disabled
      setConfig(c => ({ ...c, parallelMode: true }));
    }
  }, [config.provider]);

  const fetchModelConfig = async () => {
    try {
      const res = await fetch('/api/model/config');
      if (res.ok) {
        const data = await res.json();
        const newConfig: ModelConfig = {
          provider: data.provider || 'openrouter',
          model: data.model || 'qwen/qwen3.6-plus',
          parallelMode: data.ensemble_enabled ?? true,
        };
        setConfig(newConfig);
      }
    } catch (e) {
      console.error('Failed to fetch model config:', e);
    }
  };

  const saveModelConfig = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/model/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: config.provider,
          model: config.model,
          parallel_mode: config.parallelMode,
        }),
      });
      if (!res.ok) {
        console.error('Failed to save model config');
      }
    } catch (e) {
      console.error('Failed to save model config:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProviderChange = (provider: string) => {
    const newProvider = provider as ModelConfig['provider'];
    const models = PROVIDER_MODELS[newProvider];
    const newModel = models[0];
    const canEnsemble = !ENSEMBLE_DISABLED_PROVIDERS.includes(newProvider);
    
    const newConfig: ModelConfig = {
      provider: newProvider,
      model: newModel,
      parallelMode: canEnsemble && ENSEMBLE_ENABLED_PROVIDERS.includes(newProvider),
    };
    
    setConfig(newConfig);
    saveModelConfig();
  };

  const handleModelChange = (model: string) => {
    const newConfig = { ...config, model };
    setConfig(newConfig);
    saveModelConfig();
  };

  const handleParallelToggle = () => {
    if (ENSEMBLE_DISABLED_PROVIDERS.includes(config.provider)) {
      return; // Cannot enable for DeepSeek
    }
    const newConfig = { ...config, parallelMode: !config.parallelMode };
    setConfig(newConfig);
    saveModelConfig();
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
          {isLoading && <span className="text-amber-400">Saving...</span>}
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
              <select
                value={config.provider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="bg-navy-900 border border-white/10 rounded px-3 py-1.5 text-xs text-slate-200 focus:border-accent-teal focus:outline-none"
              >
                <option value="openrouter">OpenRouter</option>
                <option value="deepseek">DeepSeek</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
              </select>
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

            {/* Info Box */}
            <div className="flex-1 ml-4 p-2 bg-navy-900/50 rounded border border-white/5">
              <p className="text-[10px] text-slate-400 leading-relaxed">
                <span className="text-accent-teal font-medium">Ensemble Mode:</span> Runs 3 parallel analyses 
                with consensus averaging. Reduces variance in results. 
                {config.parallelMode 
                  ? ' Currently active for this ticker.' 
                  : ' Disabled - single analysis will run.'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
