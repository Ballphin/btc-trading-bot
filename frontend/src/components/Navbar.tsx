import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, History, Home, TrendingUp, Server, ServerOff } from 'lucide-react';
import { clsx } from 'clsx';
import { API_BASE_URL } from '../lib/api';

const NAV_ITEMS = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/backtest', label: 'Backtest', icon: TrendingUp },
  { to: '/history', label: 'History', icon: History },
];

export default function Navbar() {
  const location = useLocation();
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch(`${API_BASE_URL}/health`, {
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          setConnectionStatus('connected');
        } else {
          setConnectionStatus('disconnected');
        }
      } catch {
        setConnectionStatus('disconnected');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-xl bg-navy-950/80 border-b border-white/5">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 text-white font-bold text-lg no-underline">
          <Activity className="w-5 h-5 text-accent-cyan" />
          <span>TradingAgents</span>
        </Link>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            {NAV_ITEMS.map(item => {
              const isActive = location.pathname === item.to ||
                (item.to === '/history' && location.pathname.startsWith('/history')) ||
                (item.to === '/backtest' && location.pathname.startsWith('/backtest'));
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors no-underline',
                    isActive
                      ? 'bg-accent-teal/10 text-accent-teal'
                      : 'text-slate-400 hover:text-white hover:bg-white/5',
                  )}
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>
          
          {/* Connection Status */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-800/50">
            {connectionStatus === 'connected' ? (
              <>
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <Server className="w-3.5 h-3.5 text-green-400" />
                <span className="text-xs text-green-400 hidden sm:inline">Online</span>
              </>
            ) : connectionStatus === 'disconnected' ? (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500" />
                <ServerOff className="w-3.5 h-3.5 text-red-400" />
                <span className="text-xs text-red-400 hidden sm:inline">Offline</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                <Server className="w-3.5 h-3.5 text-yellow-400" />
                <span className="text-xs text-yellow-400 hidden sm:inline">Checking...</span>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
