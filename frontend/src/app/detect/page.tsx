"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Camera, CheckCircle2, ImagePlus, Loader2, RefreshCw, ScanSearch, Trash2, Upload } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ApiError, predict } from "@/lib/api";
import { useDiseases } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";
import { prepareForUpload } from "@/lib/image";
import { saveResult } from "@/lib/result-store";
import { cn } from "@/lib/utils";

const MAX_BYTES = 8 * 1024 * 1024;
const TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function DetectPage() {
  const { t, lang } = useLang();
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [camOpen, setCamOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const captureRef = useRef<HTMLInputElement>(null);
  const diseases = useDiseases(lang);
  const crops = diseases.data ? Array.from(new Set(diseases.data.map((d) => d.crop))) : [];

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const accept = useCallback((f: File | undefined | null) => {
    setError(null);
    if (!f) return;
    if (!TYPES.includes(f.type)) return setError(t("err_type"));
    if (f.size > MAX_BYTES) return setError(t("err_size"));
    setFile(f);
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
  }, [t]);

  const clear = () => { setFile(null); setPreview(null); setError(null); if (inputRef.current) inputRef.current.value = ""; };

  const openCamera = () => {
    if (typeof navigator.mediaDevices?.getUserMedia === "function") setCamOpen(true);
    else captureRef.current?.click(); // mobile fallback: native camera via <input capture>
  };

  const analyse = async () => {
    if (!file) return;
    setBusy(true); setError(null);
    try {
      const upload = await prepareForUpload(file);
      const result = await predict(upload, upload.name || "leaf.jpg", lang, true);
      await saveResult(result, file);
      router.push("/results");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("backend_down"));
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="mb-8 max-w-2xl space-y-2">
        <h1 className="text-3xl font-semibold sm:text-4xl">{t("detect_title")}</h1>
        <p className="text-muted-foreground">{t("detect_sub")}</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        <Card className="overflow-hidden p-0">
          <CardContent className="p-0">
            {camOpen ? (
              <CameraCapture
                onCancel={() => setCamOpen(false)}
                onCapture={(f) => { setCamOpen(false); accept(f); }}
                onError={() => { setCamOpen(false); setError(t("camera_error")); }}
                labels={{ capture: t("capture"), cancel: t("cancel") }}
              />
            ) : preview ? (
              <div className="relative bg-muted/40">
                {/* eslint-disable-next-line @next/next/no-img-element -- local object URL preview */}
                <img src={preview} alt="Selected leaf" className="mx-auto max-h-[480px] w-full object-contain" data-testid="preview" />
                <div className="absolute left-3 top-3 inline-flex max-w-[calc(100%-1.5rem)] items-center gap-1.5 rounded-full bg-background/90 px-3 py-1 text-xs font-medium shadow">
                  <CheckCircle2 className="size-3.5 shrink-0 text-success" /> <span className="truncate">{file?.name}</span> <span className="shrink-0">· {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB</span>
                </div>
              </div>
            ) : (
              <label
                htmlFor="leaf-file"
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => { e.preventDefault(); setDrag(false); accept(e.dataTransfer.files?.[0]); }}
                className={cn(
                  "flex min-h-[380px] cursor-pointer flex-col items-center justify-center gap-4 border-2 border-dashed border-border bg-muted/30 p-8 text-center transition-colors",
                  drag && "border-primary bg-secondary/60",
                )}
                data-testid="dropzone"
              >
                <div className="flex size-16 items-center justify-center rounded-2xl bg-secondary text-secondary-foreground">
                  <Upload className="size-7" />
                </div>
                <div>
                  <div className="text-lg font-semibold">{t("drop_title")}</div>
                  <div className="text-sm text-muted-foreground">{t("drop_hint")}</div>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2 text-sm text-muted-foreground">
                  <span className="inline-flex h-9 items-center gap-2 rounded-lg bg-primary px-4 font-medium text-primary-foreground"><ImagePlus className="size-4" />{t("browse")}</span>
                  {t("drop_or")}
                  <Button type="button" variant="outline" size="lg" className="h-9" onClick={(e) => { e.preventDefault(); openCamera(); }}>
                    <Camera /> {t("camera")}
                  </Button>
                </div>
              </label>
            )}
            <input id="leaf-file" ref={inputRef} type="file" accept={TYPES.join(",")} className="sr-only" data-testid="file-input"
              onChange={(e) => accept(e.target.files?.[0])} />
            <input ref={captureRef} type="file" accept="image/*" capture="environment" className="sr-only"
              onChange={(e) => accept(e.target.files?.[0])} />
          </CardContent>
          {preview && !camOpen && (
            <div className="flex flex-wrap items-center gap-2 border-t border-border p-4">
              <Button size="xl" onClick={analyse} disabled={busy} data-testid="analyse">
                {busy ? <Loader2 className="animate-spin" /> : <ScanSearch />} {busy ? t("analysing") : t("analyse")}
              </Button>
              <Button variant="outline" size="lg" onClick={() => inputRef.current?.click()} disabled={busy}><RefreshCw /> {t("replace")}</Button>
              <Button variant="outline" size="lg" onClick={openCamera} disabled={busy}><Camera /> {t("camera")}</Button>
              <Button variant="ghost" size="lg" onClick={clear} disabled={busy}><Trash2 /> {t("remove")}</Button>
            </div>
          )}
          {busy && <div className="h-1 w-full overflow-hidden bg-muted"><div className="h-full w-1/3 animate-[kg-progress_1.2s_ease-in-out_infinite] bg-primary" /></div>}
        </Card>

        <div className="space-y-4">
          {error && (
            <Alert variant="destructive" data-testid="detect-error">
              <AlertCircle />
              <AlertTitle>{error}</AlertTitle>
            </Alert>
          )}
          <Card>
            <CardContent className="space-y-3">
              <h2 className="text-lg font-semibold">{t("tips_title")}</h2>
              <ul className="space-y-2.5 text-sm text-muted-foreground">
                {(["tip1", "tip2", "tip3", "tip4"] as const).map((k) => (
                  <li key={k} className="flex gap-2.5"><CheckCircle2 className="mt-0.5 size-4 shrink-0 text-primary" />{t(k)}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="space-y-3">
              <h2 className="text-lg font-semibold">{t("supported_crops")}</h2>
              <div className="flex flex-wrap gap-1.5">
                {crops.map((c) => <span key={c} className="rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">{c}</span>)}
              </div>
            </CardContent>
          </Card>
          <Alert>
            <AlertCircle />
            <AlertDescription className="text-xs">{t("footer_disclaimer")}</AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
}

function CameraCapture({ onCapture, onCancel, onError, labels }: {
  onCapture: (f: File) => void; onCancel: () => void; onError: () => void; labels: { capture: string; cancel: string };
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((tr) => tr.stop());
        streamRef.current = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => onError());
    return () => { cancelled = true; streamRef.current?.getTracks().forEach((tr) => tr.stop()); };
  }, [onError]);
  const snap = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    c.toBlob((b) => b && onCapture(new File([b], `camera-${Date.now()}.jpg`, { type: "image/jpeg" })), "image/jpeg", 0.92);
  };
  return (
    <div className="relative bg-black">
      <video ref={videoRef} autoPlay playsInline muted className="mx-auto max-h-[480px] w-full object-contain" />
      <div className="pointer-events-none absolute inset-8 rounded-2xl border-2 border-white/60" />
      <div className="absolute inset-x-0 bottom-4 flex justify-center gap-3">
        <Button size="xl" onClick={snap}><Camera /> {labels.capture}</Button>
        <Button size="xl" variant="secondary" onClick={onCancel}>{labels.cancel}</Button>
      </div>
    </div>
  );
}
