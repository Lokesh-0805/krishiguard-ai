"use client";

import type { Prediction } from "./api";

// The last diagnosis is kept only in this browser tab (sessionStorage) so the results page can be
// reloaded; nothing is sent anywhere else and it disappears when the tab is closed.
const KEY = "kg-last-result";

export interface StoredResult { prediction: Prediction; image: string; createdAt: string }

async function downscale(file: Blob, max = 900): Promise<string> {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext("2d")?.drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.85);
}

export async function saveResult(prediction: Prediction, file: Blob) {
  const image = await downscale(file);
  const value: StoredResult = { prediction, image, createdAt: new Date().toISOString() };
  try { sessionStorage.setItem(KEY, JSON.stringify(value)); } catch { /* quota: results page will show empty state */ }
}

export function loadResult(): StoredResult | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as StoredResult) : null;
  } catch { return null; }
}

export function updateStoredPrediction(p: Prediction) {
  const cur = loadResult();
  if (!cur) return;
  try { sessionStorage.setItem(KEY, JSON.stringify({ ...cur, prediction: p })); } catch { /* ignore */ }
}
