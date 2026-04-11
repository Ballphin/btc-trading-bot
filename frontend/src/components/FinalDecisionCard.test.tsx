import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import FinalDecisionCard from './FinalDecisionCard';

describe('FinalDecisionCard', () => {
  it('renders structured JSON decision', () => {
    const payload = JSON.stringify({
      signal: 'BUY',
      confidence: 0.73,
      stop_loss_price: 57000,
      take_profit_price: 66000,
      max_hold_days: 7,
      reasoning: 'Momentum and breadth support upside.',
    });

    render(<FinalDecisionCard text={payload} />);

    expect(screen.getByText('Final Signal')).toBeTruthy();
    expect(screen.getByText('BUY')).toBeTruthy();
    expect(screen.getByText('Stop Loss')).toBeTruthy();
    expect(screen.getByText('Take Profit')).toBeTruthy();
    expect(screen.getByText('Reasoning')).toBeTruthy();
  });

  it('falls back to markdown text', () => {
    render(<FinalDecisionCard text={'## Plain Decision\nHold until confirmation.'} />);
    expect(screen.getByText('Plain Decision')).toBeTruthy();
    expect(screen.getByText('Hold until confirmation.')).toBeTruthy();
  });
});
