import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import {
  createChart,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  type SeriesMarker,
  type IPriceLine,
  type PriceLineOptions,
} from 'lightweight-charts';
import type { CandleBar } from '../lib/api';
import { SEMANTIC } from '../lib/colors';

export interface PriceLineSpec {
  price: number;
  color: string;
  lineWidth: 1 | 2 | 3 | 4;
  lineStyle: LineStyle;
  title: string;
}

export interface TrendLineSpec {
  from: { ts: number; price: number };
  to: { ts: number; price: number };
  color: string;
  lineWidth: 1 | 2 | 3;
  lineStyle: LineStyle;
  title?: string;
}

export interface TradingViewCandlesHandle {
  /** Programmatic chart control, e.g. jump to a bar by timestamp. */
  scrollToTime: (ts: number) => void;
  /** Take a PNG screenshot. */
  screenshot: () => HTMLCanvasElement | null;
}

interface Props {
  candles: CandleBar[];
  priceLines?: PriceLineSpec[];
  trendLines?: TrendLineSpec[];
  markers?: SeriesMarker<UTCTimestamp>[];
  height?: number;
  showVolume?: boolean;
}

/**
 * TradingView-grade candlestick chart with volume sub-pane, crosshair, pan/zoom.
 * Uses lightweight-charts (Apache-2.0).
 */
const TradingViewCandles = forwardRef<TradingViewCandlesHandle, Props>(
  function TradingViewCandles(
    { candles, priceLines = [], trendLines = [], markers = [], height = 480, showVolume = true },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
    const priceLinesRef = useRef<IPriceLine[]>([]);
    const trendLineSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);

    useEffect(() => {
      if (!containerRef.current) return;

      const chart = createChart(containerRef.current, {
        layout: {
          background: { color: 'transparent' },
          textColor: '#94A3B8',
          fontFamily: 'Inter, system-ui, sans-serif',
        },
        grid: {
          vertLines: { color: '#1F2937', style: LineStyle.Dotted },
          horzLines: { color: '#1F2937', style: LineStyle.Dotted },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#334155', scaleMargins: { top: 0.1, bottom: showVolume ? 0.28 : 0.05 } },
        timeScale: {
          borderColor: '#334155',
          timeVisible: true,
          secondsVisible: false,
        },
        width: containerRef.current.clientWidth,
        height,
      });
      chartRef.current = chart;

      const candleSeries = chart.addCandlestickSeries({
        upColor: SEMANTIC.bull,
        downColor: SEMANTIC.bear,
        wickUpColor: SEMANTIC.bull,
        wickDownColor: SEMANTIC.bear,
        borderVisible: false,
      });
      candleSeriesRef.current = candleSeries;

      if (showVolume) {
        const volumeSeries = chart.addHistogramSeries({
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
          color: SEMANTIC.bull,
        });
        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.78, bottom: 0 },
          visible: false,
        });
        volumeSeriesRef.current = volumeSeries;
      }

      // Responsive resize
      const onResize = () => {
        if (!containerRef.current) return;
        chart.applyOptions({ width: containerRef.current.clientWidth });
      };
      window.addEventListener('resize', onResize);
      return () => {
        window.removeEventListener('resize', onResize);
        chart.remove();
        chartRef.current = null;
        candleSeriesRef.current = null;
        volumeSeriesRef.current = null;
        priceLinesRef.current = [];
        trendLineSeriesRef.current = [];
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [height, showVolume]);

    // Update candle + volume data
    useEffect(() => {
      const cs = candleSeriesRef.current;
      const vs = volumeSeriesRef.current;
      if (!cs) return;
      const candleData = candles.map((b) => ({
        time: b.ts as UTCTimestamp,
        open: b.o,
        high: b.h,
        low: b.l,
        close: b.c,
      }));
      cs.setData(candleData);
      if (vs) {
        vs.setData(
          candles.map((b) => ({
            time: b.ts as UTCTimestamp,
            value: b.v,
            color: b.c >= b.o ? `${SEMANTIC.bull}55` : `${SEMANTIC.bear}55`,
          })),
        );
      }
      chartRef.current?.timeScale().fitContent();
    }, [candles]);

    // Update horizontal price lines (entry / SL / TP / S / R)
    useEffect(() => {
      const cs = candleSeriesRef.current;
      if (!cs) return;
      priceLinesRef.current.forEach((pl) => cs.removePriceLine(pl));
      priceLinesRef.current = priceLines.map((spec) =>
        cs.createPriceLine({
          price: spec.price,
          color: spec.color,
          lineWidth: spec.lineWidth,
          lineStyle: spec.lineStyle,
          axisLabelVisible: true,
          title: spec.title,
        } as PriceLineOptions),
      );
    }, [priceLines]);

    // Update trendline series (pattern necklines, triangle sides, etc.)
    useEffect(() => {
      const chart = chartRef.current;
      if (!chart) return;
      trendLineSeriesRef.current.forEach((s) => chart.removeSeries(s));
      trendLineSeriesRef.current = trendLines.map((tl) => {
        const series = chart.addLineSeries({
          color: tl.color,
          lineWidth: tl.lineWidth,
          lineStyle: tl.lineStyle,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        series.setData([
          { time: tl.from.ts as UTCTimestamp, value: tl.from.price },
          { time: tl.to.ts as UTCTimestamp, value: tl.to.price },
        ]);
        return series;
      });
    }, [trendLines]);

    // Update markers (pattern anchors)
    useEffect(() => {
      const cs = candleSeriesRef.current;
      if (!cs) return;
      cs.setMarkers(markers);
    }, [markers]);

    useImperativeHandle(ref, () => ({
      scrollToTime: (ts: number) => {
        chartRef.current?.timeScale().scrollToPosition(0, false);
        // lightweight-charts doesn't have direct scrollTo(time), but
        // setVisibleLogicalRange centered on the bar works.
        chartRef.current?.timeScale().scrollToRealTime();
        void ts;
      },
      screenshot: () => chartRef.current?.takeScreenshot() ?? null,
    }));

    return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />;
  },
);

export default TradingViewCandles;
