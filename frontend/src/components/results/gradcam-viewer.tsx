"use client";

import { useState } from "react";
import { Info } from "lucide-react";
import type { Explanation } from "@/lib/api";
import { useLang } from "@/lib/i18n";
import { cn } from "@/lib/utils";

const src = (b64: string) => `data:image/png;base64,${b64}`;

export function GradCamViewer({ e, labels }: { e: Explanation; labels: { original: string; heatmap: string; overlay: string } }) {
  const { t } = useLang();
  const [mode, setMode] = useState<"compare" | "grid">("compare");
  const [split, setSplit] = useState(50);
  return (
    <div className="space-y-4" data-testid="gradcam">
      <div className="inline-flex rounded-lg bg-muted p-1 text-xs font-medium">
        {(["compare", "grid"] as const).map((m) => (
          <button key={m} onClick={() => setMode(m)} className={cn("rounded-md px-3 py-1.5 capitalize", mode === m && "bg-background shadow-sm")}>
            {m === "compare" ? `${labels.original} ↔ ${labels.overlay}` : `${labels.original} · ${labels.heatmap} · ${labels.overlay}`}
          </button>
        ))}
      </div>
      {mode === "compare" ? (
        <div className="space-y-2">
          <div className="relative mx-auto aspect-square w-full max-w-sm overflow-hidden rounded-xl border border-border">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src(e.overlay_png)} alt={labels.overlay} className="absolute inset-0 size-full object-cover [image-rendering:auto]" />
            <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src(e.model_input_png)} alt={labels.original} className="size-full object-cover" />
            </div>
            <div className="absolute inset-y-0 w-0.5 bg-white shadow" style={{ left: `${split}%` }} />
          </div>
          <input type="range" min={0} max={100} value={split} onChange={(ev) => setSplit(Number(ev.target.value))}
            className="mx-auto block w-full max-w-sm accent-[var(--primary)]" aria-label="Compare original and overlay" />
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-2">
          {([[e.model_input_png, labels.original], [e.heatmap_png, labels.heatmap], [e.overlay_png, labels.overlay]] as const).map(([b, l]) => (
            <figure key={l} className="space-y-1">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src(b)} alt={l} className="aspect-square w-full rounded-lg border border-border object-cover" />
              <figcaption className="text-center text-xs text-muted-foreground">{l}</figcaption>
            </figure>
          ))}
        </div>
      )}
      <p className="flex gap-2 text-xs text-muted-foreground"><Info className="size-3.5 shrink-0" /><span>{t("gradcam_note")} ({e.method}, layer <code>{e.target_layer}</code>)</span></p>
    </div>
  );
}
