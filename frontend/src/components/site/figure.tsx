"use client";

import { useState } from "react";

// Figures are real artefacts copied from ml/artifacts by scripts/sync_figures.sh; hidden if absent.
export function Figure({ src, caption }: { src: string; caption: string }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <figure className="overflow-hidden rounded-2xl border border-border bg-white">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={caption} className="w-full" loading="lazy" onError={() => setOk(false)} />
      <figcaption className="border-t border-border bg-card px-4 py-2 text-xs text-muted-foreground">{caption}</figcaption>
    </figure>
  );
}
