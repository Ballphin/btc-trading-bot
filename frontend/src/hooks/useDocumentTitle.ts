import { useEffect } from 'react';

export default function useDocumentTitle(title: string) {
  useEffect(() => {
    const prev = document.title;
    document.title = title ? `${title} — TradingAgents` : 'TradingAgents';
    return () => { document.title = prev; };
  }, [title]);
}
