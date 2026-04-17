import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import ErrorBoundary from './components/ErrorBoundary';

const Home = lazy(() => import('./pages/Home'));
const Analyze = lazy(() => import('./pages/Analyze'));
const History = lazy(() => import('./pages/History'));
const AnalysisDetail = lazy(() => import('./pages/AnalysisDetail'));
const Backtest = lazy(() => import('./pages/Backtest'));
const BacktestResults = lazy(() => import('./pages/BacktestResults'));
const RecentBacktests = lazy(() => import('./pages/RecentBacktests'));
const Scorecard = lazy(() => import('./pages/Scorecard'));
const Pulse = lazy(() => import('./pages/Pulse'));

function PageLoader() {
  return (
    <div className="flex items-center justify-center py-32">
      <div className="w-8 h-8 border-2 border-accent-teal/30 border-t-accent-teal rounded-full animate-spin" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-navy-950">
        <Navbar />
        <main>
          <ErrorBoundary>
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/analyze/:ticker" element={<ErrorBoundary><Analyze /></ErrorBoundary>} />
                <Route path="/history" element={<History />} />
                <Route path="/history/:ticker" element={<History />} />
                <Route path="/history/:ticker/:date" element={<ErrorBoundary><AnalysisDetail /></ErrorBoundary>} />
                <Route path="/backtest" element={<Backtest />} />
                <Route path="/backtest/results/:id" element={<ErrorBoundary><BacktestResults /></ErrorBoundary>} />
                <Route path="/backtest/:jobId" element={<ErrorBoundary><BacktestResults /></ErrorBoundary>} />
                <Route path="/backtests" element={<RecentBacktests />} />
                <Route path="/scorecard" element={<Scorecard />} />
                <Route path="/pulse" element={<Pulse />} />
              </Routes>
            </Suspense>
          </ErrorBoundary>
        </main>
      </div>
    </BrowserRouter>
  );
}
