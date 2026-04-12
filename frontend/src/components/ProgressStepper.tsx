import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Loader2, Circle, AlertCircle, Wifi, WifiOff } from 'lucide-react';
import { clsx } from 'clsx';

export interface StepState {
  key: string;
  label: string;
  status: 'pending' | 'running' | 'done' | 'error';
  content?: string;
}

interface Props {
  steps: StepState[];
  currentStep: number;
  totalSteps: number;
  isRunning?: boolean;
  lastHeartbeat?: number;
  startTime?: number;
}

function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min === 0) return `${sec}s`;
  return `${min}m ${sec}s`;
}

export default function ProgressStepper({ steps, currentStep, totalSteps, isRunning, lastHeartbeat, startTime }: Props) {
  const pct = totalSteps > 0 ? Math.round((currentStep / totalSteps) * 100) : 0;
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  const elapsed = startTime ? now - startTime : 0;
  const sinceBeat = lastHeartbeat ? now - lastHeartbeat : Infinity;

  // Connection health: green < 30s, amber 30-120s, red > 120s
  let connStatus: 'alive' | 'stale' | 'lost' = 'alive';
  if (sinceBeat > 120_000) connStatus = 'lost';
  else if (sinceBeat > 30_000) connStatus = 'stale';

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div className="glass p-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-slate-400">Analysis Progress</span>
          <span className="text-accent-cyan font-mono font-bold">{pct}%</span>
        </div>
        <div className="w-full h-2 bg-navy-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-accent-teal to-accent-cyan"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>

        {/* Connection status badge */}
        {isRunning && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/5">
            <div className="flex items-center gap-2">
              {connStatus === 'alive' && (
                <>
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
                  </span>
                  <span className="text-xs text-emerald-400 font-medium">AI Processing</span>
                </>
              )}
              {connStatus === 'stale' && (
                <>
                  <Wifi className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-xs text-amber-400 font-medium">Waiting for response...</span>
                </>
              )}
              {connStatus === 'lost' && (
                <>
                  <WifiOff className="w-3.5 h-3.5 text-red-400" />
                  <span className="text-xs text-red-400 font-medium">Connection may be lost</span>
                </>
              )}
            </div>
            <span className="text-xs text-slate-500 font-mono">{formatElapsed(elapsed)}</span>
          </div>
        )}
      </div>

      {/* Steps */}
      <div className="space-y-1">
        {steps.map((step, i) => (
          <motion.div
            key={step.key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className={clsx(
              'flex items-center gap-3 p-3 rounded-lg transition-colors',
              step.status === 'running' && 'bg-accent-teal/5 border border-accent-teal/20',
              step.status === 'done' && 'opacity-80',
              step.status === 'pending' && 'opacity-40',
            )}
          >
            {/* Icon */}
            {step.status === 'done' && <CheckCircle className="w-5 h-5 text-accent-cyan shrink-0" />}
            {step.status === 'running' && <Loader2 className="w-5 h-5 text-accent-teal shrink-0 animate-spin" />}
            {step.status === 'error' && <AlertCircle className="w-5 h-5 text-accent-red shrink-0" />}
            {step.status === 'pending' && <Circle className="w-5 h-5 text-slate-600 shrink-0" />}

            {/* Label */}
            <span className={clsx(
              'text-sm font-medium',
              step.status === 'running' && 'text-white',
              step.status === 'done' && 'text-slate-300',
              step.status === 'pending' && 'text-slate-500',
            )}>
              {step.label}
            </span>

            {/* Status text */}
            {step.status === 'running' && (
              <span className="ml-auto text-xs text-accent-teal animate-pulse">Processing...</span>
            )}
            {step.status === 'done' && (
              <span className="ml-auto text-xs text-slate-500">Complete</span>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
