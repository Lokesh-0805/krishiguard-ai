"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle, CheckCircle2, CircleHelp, Download, ExternalLink, FlaskConical, Leaf, Printer, ScanSearch, ShieldCheck, Sprout, XCircle,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { GradCamViewer } from "@/components/results/gradcam-viewer";
import { FeedbackPanel } from "@/components/results/feedback";
import { AdvisoryBody } from "@/components/advisory/advisory-body";
import { getAdvisory, type Advisory, type Status } from "@/lib/api";
import { pct, useDiseases } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";
import { buildReportHtml } from "@/lib/report";
import { loadResult, type StoredResult } from "@/lib/result-store";
import { cn } from "@/lib/utils";

const STATUS_MSG = {
  en: {
    confident: "The model is confident about this result. Confirm with a local expert before spraying.",
    uncertain: "The model is not sure. Possible matches are shown below — retake a clear photo of a single leaf or consult your local KVK.",
    unrecognized: "This image could not be matched to a supported crop or disease. It may be an unsupported plant, not a leaf, or a poor-quality photo.",
  },
  hi: {
    confident: "मॉडल इस परिणाम को लेकर आश्वस्त है। छिड़काव से पहले स्थानीय विशेषज्ञ से पुष्टि करें।",
    uncertain: "मॉडल निश्चित नहीं है। संभावित परिणाम नीचे दिए हैं — एक पत्ती की साफ़ फ़ोटो दोबारा लें या अपने नज़दीकी KVK से सलाह लें।",
    unrecognized: "यह फ़ोटो किसी समर्थित फसल या रोग से मेल नहीं खाती। यह असमर्थित पौधा हो सकता है, पत्ती न हो, या फ़ोटो साफ़ न हो।",
  },
} as const;

const STATUS_STYLE: Record<Status, { icon: typeof CheckCircle2; cls: string }> = {
  confident: { icon: ShieldCheck, cls: "border-success/40 bg-success/10 text-foreground" },
  uncertain: { icon: CircleHelp, cls: "border-warning/50 bg-warning/15 text-foreground" },
  unrecognized: { icon: XCircle, cls: "border-destructive/40 bg-destructive/10 text-foreground" },
};

export default function ResultsPage() {
  const { t, lang } = useLang();
  const [stored, setStored] = useState<StoredResult | null | undefined>(undefined);
  const [advisory, setAdvisory] = useState<Advisory | null>(null);
  const [advLoading, setAdvLoading] = useState(false);
  const diseases = useDiseases(lang);

  // eslint-disable-next-line react-hooks/set-state-in-effect -- sessionStorage is only readable after mount
  useEffect(() => setStored(loadResult()), []);
  const p = stored?.prediction;

  // advisory is fetched in the selected language (only for confident results)
  // advice belongs to one prediction: drop it when a different diagnosis is shown
  const [advFor, setAdvFor] = useState<string | null>(null);
  useEffect(() => {
    if (!p || p.status !== "confident") return;
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for the language switch
    setAdvLoading(true);
    getAdvisory(p.class_id, lang)
      .then((a) => { if (alive) { setAdvisory(a); setAdvFor(p.prediction_id ?? p.class_id); } })
      .catch(() => { if (alive) { setAdvisory(p.advisory && p.advisory.language === lang ? p.advisory : null); setAdvFor(p.prediction_id ?? p.class_id); } })
      .finally(() => alive && setAdvLoading(false));
    return () => { alive = false; };
  }, [p, lang]);

  const names = useMemo(() => new Map((diseases.data ?? []).map((d) => [d.id, d])), [diseases.data]);
  const label = (id: string, fallback: { crop: string; disease: string }) => {
    const d = names.get(id);
    return d ? { crop: d.crop, disease: d.name, slug: d.slug } : { ...fallback, slug: id };
  };

  const adv = p && advFor === (p.prediction_id ?? p.class_id) ? advisory : null;

  if (stored === undefined) {
    return <div className="mx-auto max-w-6xl px-4 py-14"><Skeleton className="h-96 w-full rounded-2xl" /></div>;
  }
  if (!p || !stored) {
    return (
      <div className="mx-auto flex max-w-xl flex-col items-center gap-4 px-4 py-24 text-center">
        <Leaf className="size-12 text-muted-foreground" />
        <h1 className="text-2xl font-semibold">{t("no_result")}</h1>
        <Button asChild size="xl"><Link href="/detect"><ScanSearch /> {t("go_detect")}</Link></Button>
      </div>
    );
  }

  const main = label(p.class_id, { crop: p.crop, disease: p.disease });
  const S = STATUS_STYLE[p.status];
  const statusTitle = t(p.status === "confident" ? "status_confident" : p.status === "uncertain" ? "status_uncertain" : "status_unrecognized");
  const showDiagnosis = p.status !== "unrecognized";

  const download = () => {
    const html = buildReportHtml({ stored, advisory: p.status === "confident" ? adv : null, lang, names: main, statusTitle, statusMsg: STATUS_MSG[lang][p.status] });
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `krishiguard-report-${stored.createdAt.slice(0, 19).replace(/[:T]/g, "-")}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-12" data-testid="results">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted-foreground">{new Date(stored.createdAt).toLocaleString(lang === "hi" ? "hi-IN" : "en-IN")} · {p.model_version}</p>
          <h1 className="text-3xl font-semibold sm:text-4xl">{t("result_title")}</h1>
        </div>
        <div className="no-print flex flex-wrap gap-2">
          <Button variant="outline" onClick={download} data-testid="download-report"><Download /> {t("download_report")}</Button>
          <Button variant="outline" onClick={() => window.print()}><Printer /> {t("print")}</Button>
          <Button asChild><Link href="/detect"><ScanSearch /> {t("new_diagnosis")}</Link></Button>
        </div>
      </div>

      <div className={cn("mb-6 flex gap-3 rounded-2xl border p-4", S.cls)} data-testid="status-banner" data-status={p.status}>
        <S.icon className="mt-0.5 size-5 shrink-0" />
        <div>
          <div className="font-semibold">{statusTitle}</div>
          <p className="text-sm text-muted-foreground">{STATUS_MSG[lang][p.status]}</p>
        </div>
      </div>

      {p.quality.flags.length > 0 && (
        <Alert className="mb-6 border-warning/50">
          <AlertTriangle className="text-warning" />
          <AlertTitle>
            {p.quality.flags.map((f) => (f === "blurry" ? t("quality_blurry") : f === "low_resolution" ? t("quality_lowres") : f)).join(" ")}
          </AlertTitle>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_1.15fr]">
        <div className="space-y-6">
          <Card className="overflow-hidden p-0">
            {/* eslint-disable-next-line @next/next/no-img-element -- data URL kept in session */}
            <img src={stored.image} alt="Uploaded leaf" className="max-h-[420px] w-full bg-muted object-contain" />
          </Card>
          {p.explanation && (
            <Card>
              <CardHeader><CardTitle className="text-lg">{t("explain_title")}</CardTitle></CardHeader>
              <CardContent>
                <GradCamViewer e={p.explanation} labels={{ original: t("original"), heatmap: t("heatmap"), overlay: t("overlay") }} />
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card data-testid="diagnosis-card">
            <CardContent className="space-y-6">
              {showDiagnosis ? (
                <>
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="space-y-1">
                      <div className="text-sm font-medium text-muted-foreground">{t("crop")}: <span className="text-foreground" data-testid="crop-name">{main.crop}</span></div>
                      <div className="font-heading text-3xl font-semibold leading-tight" data-testid="disease-name">{main.disease}</div>
                      {adv?.scientific_name && <div className="text-sm italic text-muted-foreground">{adv.scientific_name}</div>}
                    </div>
                    <Badge className={cn("gap-1 px-3 py-1 text-sm", p.healthy ? "bg-success/15 text-foreground" : "bg-destructive/12 text-foreground")} variant="outline">
                      {p.healthy ? <Leaf className="size-3.5 text-success" /> : <AlertTriangle className="size-3.5 text-destructive" />}
                      {p.healthy ? t("healthy") : t("diseased")}
                    </Badge>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Meter label={t("confidence")} value={p.confidence} tone={p.status} testId="confidence" />
                    <Meter label={t("crop_conf")} value={p.crop_confidence} tone={p.crop_confidence >= 0.8 ? "confident" : "uncertain"} />
                  </div>
                </>
              ) : null}
              <div>
                <div className="mb-2 text-sm font-semibold">{t("other_candidates")}</div>
                <ul className="space-y-2" data-testid="top-k">
                  {p.top_k.map((k, i) => {
                    const l = label(k.class_id, k);
                    return (
                      <li key={k.class_id} className="flex items-center gap-3 text-sm">
                        <span className="w-5 text-muted-foreground">{i + 1}.</span>
                        <div className="min-w-0 flex-1">
                          <div className="flex justify-between gap-2"><span className="truncate">{l.crop} — {l.disease}</span><span className="tabular-nums text-muted-foreground">{pct(k.probability)}</span></div>
                          <div className="mt-1 h-1.5 rounded-full bg-muted"><div className="h-full rounded-full bg-primary/70" style={{ width: `${Math.max(2, k.probability * 100)}%` }} /></div>
                        </div>
                        {p.status !== "confident" && (
                          <Link href={`/diseases#id=${l.slug}`} className="no-print shrink-0 text-xs font-medium text-primary hover:underline">{t("view_advisory")}</Link>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            </CardContent>
          </Card>

          {p.status === "confident" ? (
            advLoading && !adv ? <Skeleton className="h-72 w-full rounded-2xl" /> : adv && (
              <Card data-testid="advisory">
                <CardContent>
                  <Tabs defaultValue="overview">
                    <TabsList className="mb-4 flex h-auto w-full flex-wrap justify-start">
                      <TabsTrigger value="overview"><Leaf />{t("overview")}</TabsTrigger>
                      <TabsTrigger value="symptoms">{t("symptoms")}</TabsTrigger>
                      {!adv.healthy && <TabsTrigger value="treatment"><FlaskConical />{t("treatment")}</TabsTrigger>}
                      <TabsTrigger value="prevention"><Sprout />{t("prevention")}</TabsTrigger>
                      <TabsTrigger value="sources"><ExternalLink />{t("sources")}</TabsTrigger>
                    </TabsList>
                    <AdvisoryBody a={adv} mode="tabs" />
                  </Tabs>
                </CardContent>
              </Card>
            )
          ) : (
            <Alert className="border-warning/50"><CircleHelp className="text-warning" /><AlertDescription>{t("advisory_hidden")}</AlertDescription></Alert>
          )}

          {p.prediction_id && <FeedbackPanel key={p.prediction_id} predictionId={p.prediction_id} />}
        </div>
      </div>
    </div>
  );
}

function Meter({ label, value, tone, testId }: { label: string; value: number; tone: Status; testId?: string }) {
  const color = tone === "confident" ? "bg-success" : tone === "uncertain" ? "bg-warning" : "bg-destructive";
  return (
    <div className="rounded-xl border border-border p-4">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="font-heading text-3xl font-semibold tabular-nums" data-testid={testId}>{pct(value)}</div>
      <div className="mt-2 h-2 rounded-full bg-muted"><div className={cn("h-full rounded-full", color)} style={{ width: `${value * 100}%` }} /></div>
    </div>
  );
}
