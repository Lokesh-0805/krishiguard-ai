"use client";

import { useEffect, useState } from "react";
import { getMetrics, listDiseases, type DiseaseSummary, type Lang, type MetricsSummary } from "./api";

type State<T> = { data: T | null; error: string | null; loading: boolean };

let metricsPromise: Promise<MetricsSummary> | null = null;

export function useMetrics(): State<MetricsSummary> & { reload: () => void } {
  const [s, setS] = useState<State<MetricsSummary>>({ data: null, error: null, loading: true });
  const [n, setN] = useState(0);
  useEffect(() => {
    let alive = true;
    metricsPromise ??= getMetrics();
    metricsPromise
      .then((d) => alive && setS({ data: d, error: null, loading: false }))
      .catch((e: Error) => {
        metricsPromise = null;
        if (alive) setS({ data: null, error: e.message, loading: false });
      });
    return () => { alive = false; };
  }, [n]);
  return { ...s, reload: () => { setS({ data: null, error: null, loading: true }); setN((x) => x + 1); } };
}

export function useDiseases(lang: Lang): State<DiseaseSummary[]> {
  const [s, setS] = useState<State<DiseaseSummary[]>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    listDiseases(lang)
      .then((d) => alive && setS({ data: d.items, error: null, loading: false }))
      .catch((e: Error) => alive && setS({ data: null, error: e.message, loading: false }));
    return () => { alive = false; };
  }, [lang]);
  return s;
}

export const pct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
