import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import History from './pages/History';
import AnalysisDetail from './pages/AnalysisDetail';
import Backtest from './pages/Backtest';
import BacktestResults from './pages/BacktestResults';
import RecentBacktests from './pages/RecentBacktests';
import ErrorBoundary from './components/ErrorBoundary';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-navy-950">
        <Navbar />
        <ErrorBoundary>
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
          </Routes>
        </ErrorBoundary>
      </div>
    </BrowserRouter>
  );
}
