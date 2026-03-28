import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import History from './pages/History';
import AnalysisDetail from './pages/AnalysisDetail';
import Backtest from './pages/Backtest';
import BacktestResults from './pages/BacktestResults';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-navy-950">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze/:ticker" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:ticker" element={<History />} />
          <Route path="/history/:ticker/:date" element={<AnalysisDetail />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/backtest/:jobId" element={<BacktestResults />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
