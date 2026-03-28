import { motion } from 'framer-motion';
import { CheckCircle, Loader2, Circle, AlertCircle } from 'lucide-react';
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
}

export default function ProgressStepper({ steps, currentStep, totalSteps }: Props) {
  const pct = totalSteps > 0 ? Math.round((currentStep / totalSteps) * 100) : 0;

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
