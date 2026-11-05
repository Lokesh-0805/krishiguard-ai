"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, Award, Cpu, Gauge, Info, RefreshCw, Target } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { prettyClass, type MetricsSummary } from "@/lib/api";
import { pct, useMetrics } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";
import { cn } from "@/lib/utils";

// Evaluation sets keep the same colour slot on every chart (colour follows the entity).
const SETS = [
  { key: "test", label: "PlantVillage test (lab)", color: "var(--series-1)" },
  { key: "external_test_plantdoc", label: "PlantDoc test (field)", color: "var(--series-2)" },
  { key: "external_test_tomato_pk_field", label: "Tomato field (PK)", color: "var(--series-3)" },
] as const;
type SetKey = (typeof SETS)[number]["key"];
const MODEL_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

const axis = { stroke: "var(--muted-foreground)", fontSize: 11, tickLine: false };
const tooltipStyle = {
  contentStyle: { background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, color: "var(--popover-foreground)" },
  cursor: { fill: "var(--muted)", opacity: 0.5 },
};

export default function PerformancePage() {
  const { t } = useLang();
  const { data, error, loading, reload } = useMetrics();

  if (loading) return <Shell t={t}><Skeleton className="h-40 w-full rounded-2xl" /><Skeleton className="h-96 w-full rounded-2xl" /></Shell>;
  if (error || !data)
    return (
      <Shell t={t}>
        <Alert variant="destructive"><AlertTriangle /><AlertTitle>{t("backend_down")}</AlertTitle>
          <AlertDescription>{error} <Button variant="outline" size="sm" className="ml-2" onClick={reload}><RefreshCw />{t("retry")}</Button></AlertDescription>
        </Alert>
      </Shell>
    );
  return <Shell t={t}><Dashboard m={data} /></Shell>;
}

function Shell({ t, children }: { t: (k: "perf_title" | "perf_sub") => string; children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6 sm:py-14" data-testid="performance">
      <div className="max-w-3xl space-y-2">
        <h1 className="text-3xl font-semibold sm:text-4xl">{t("perf_title")}</h1>
        <p className="text-muted-foreground">{t("perf_sub")}</p>
      </div>
      {children}
    </div>
  );
}

function Dashboard({ m }: { m: MetricsSummary }) {
  const sel = m.models.find((x) => x.selected)!;
  const test = sel.results.test;
  const [metric, setMetric] = useState<"macro_f1" | "accuracy">("macro_f1");
  const [pcSet, setPcSet] = useState<SetKey>("test");

  const comparison = m.models.map((r) => ({
    name: r.display_name, ...Object.fromEntries(SETS.map((s) => [s.key, +(r.results[s.key]?.[metric] ?? 0).toFixed(4)])),
  }));
  const rc = useMemo(() => {
    const rows: Record<string, number | null>[] = [];
    const grid = Array.from({ length: 21 }, (_, i) => i / 20);
    for (const c of grid) {
      const row: Record<string, number | null> = { coverage: c };
      for (const s of SETS) {
        const curve = m.selected_detail.risk_coverage[s.key];
        if (!curve) continue;
        let idx = curve.coverage.findIndex((x) => x >= c);
        if (idx < 0) idx = curve.coverage.length - 1;
        row[s.key] = c === 0 ? null : curve.selective_accuracy[idx];
      }
      rows.push(row);
    }
    return rows;
  }, [m]);
  const maxEpochs = Math.max(...m.models.map((r) => r.history.length));
  const curves = Array.from({ length: maxEpochs }, (_, i) => ({
    epoch: i + 1, ...Object.fromEntries(m.models.flatMap((r) => [[`${r.run}:val`, r.history[i]?.val_macro_f1 ?? null], [`${r.run}:ext`, r.history[i]?.extval_macro_f1 ?? null]])),
  }));

  return (
    <>
      <Card className="border-primary/30 bg-secondary/30">
        <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <span className="flex items-center gap-2 font-semibold"><Award className="size-4 text-primary" />Deployed: {sel.display_name}</span>
          <span className="text-muted-foreground">{sel.parameters_m.toFixed(1)}M params · {sel.img_size}px · {sel.checkpoint_mb.toFixed(0)} MB checkpoint</span>
          <span className="text-muted-foreground">T = {m.serving.temperature.toFixed(2)} · confident ≥ {pct(m.serving.confidence_threshold)}</span>
          <span className="w-full text-xs text-muted-foreground"><Info className="mr-1 inline size-3.5" />{m.selection.rule}</span>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5" data-testid="kpis">
        <Kpi icon={Target} label="Accuracy" value={pct(test.accuracy, 2)} note="PlantVillage test" />
        <Kpi icon={Gauge} label="Macro precision" value={pct(test.macro_precision, 2)} note="PlantVillage test" />
        <Kpi icon={Gauge} label="Macro recall" value={pct(test.macro_recall, 2)} note="PlantVillage test" />
        <Kpi icon={Award} label="Macro-F1" value={pct(test.macro_f1, 2)} note={`n = ${test.n.toLocaleString()}`} />
        <Kpi icon={Cpu} label="CPU latency" value={`${sel.latency.cpu.batch1_ms_mean.toFixed(0)} ms`} note="batch 1, mean" />
      </div>

      <Alert className="border-warning/50 bg-warning/10">
        <AlertTriangle className="text-warning" />
        <AlertTitle>Lab accuracy does not transfer to field photos</AlertTitle>
        <AlertDescription>
          On real field photos the same model scores {pct(sel.results.external_test_plantdoc?.accuracy)} (PlantDoc) and{" "}
          {pct(sel.results.external_test_tomato_pk_field?.accuracy)} (tomato field set) accuracy, versus {pct(test.accuracy)} on
          PlantVillage-style lab photos. Crop identification on PlantDoc is {pct(sel.results.external_test_plantdoc?.crop_accuracy)}.
          This is why uncertain predictions are flagged and advice is withheld below the confidence threshold.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader className="flex flex-row items-start justify-between gap-4">
            <div><CardTitle>Lab vs field generalisation</CardTitle><CardDescription>Same model, held-out lab photos vs real field photos</CardDescription></div>
            <Tabs value={metric} onValueChange={(v) => setMetric(v as typeof metric)}>
              <TabsList><TabsTrigger value="macro_f1">Macro-F1</TabsTrigger><TabsTrigger value="accuracy">Accuracy</TabsTrigger></TabsList>
            </Tabs>
          </CardHeader>
          <CardContent>
            <div className="h-80" data-testid="comparison-chart">
              <ResponsiveContainer>
                <BarChart data={comparison} barGap={2} margin={{ left: -10, right: 8 }}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="name" {...axis} interval={0} tick={{ fontSize: 11 }} />
                  <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip {...tooltipStyle} formatter={(v) => pct(Number(v))} />
                  <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />
                  {SETS.map((s) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Risk–coverage</CardTitle><CardDescription>Accuracy on the images the deployed model answers, as the confidence bar is raised</CardDescription></CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer>
                <LineChart data={rc} margin={{ left: -10, right: 12 }}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="coverage" {...axis} type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip {...tooltipStyle} formatter={(v) => pct(Number(v))} labelFormatter={(v) => `coverage ${pct(Number(v), 0)}`} />
                  <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                  {SETS.map((s) => <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />)}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Model summary</CardTitle><CardDescription>Every number is computed from saved predictions on held-out data</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <Table data-testid="model-table">
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead><TableHead>Trained on</TableHead><TableHead className="text-right">Params</TableHead>
                <TableHead className="text-right">Epochs</TableHead><TableHead className="text-right">Val F1</TableHead><TableHead className="text-right">Ext-val F1</TableHead>
                {SETS.map((s) => <TableHead key={s.key} className="text-right">{s.label} acc / F1</TableHead>)}
                <TableHead className="text-right">CPU ms</TableHead><TableHead className="text-right">ECE (cal.)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {m.models.map((r) => (
                <TableRow key={r.run} className={cn(r.selected && "bg-secondary/40")}>
                  <TableCell className="font-medium">{r.display_name} {r.selected && <Badge className="ml-1">deployed</Badge>}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{r.trained_on}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.parameters_m.toFixed(1)}M</TableCell>
                  <TableCell className="text-right tabular-nums">{r.epochs_trained} (best {r.best_epoch})</TableCell>
                  <TableCell className="text-right tabular-nums">{pct(r.results.validation.macro_f1, 2)}</TableCell>
                  <TableCell className="text-right tabular-nums">{pct(r.results.external_validation.macro_f1, 1)}</TableCell>
                  {SETS.map((s) => <TableCell key={s.key} className="text-right tabular-nums">{pct(r.results[s.key]?.accuracy)} / {pct(r.results[s.key]?.macro_f1)}</TableCell>)}
                  <TableCell className="text-right tabular-nums">{r.latency.cpu.batch1_ms_mean.toFixed(0)}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.results.test.ece_calibrated.toFixed(3)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        {(["val", "ext"] as const).map((k) => (
          <Card key={k}>
            <CardHeader>
              <CardTitle>{k === "val" ? "Validation macro-F1 per epoch" : "External-validation (PlantDoc) macro-F1 per epoch"}</CardTitle>
              <CardDescription>{k === "val" ? "PlantVillage validation split" : "Held-out PlantDoc images used only for calibration and selection"}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer>
                  <LineChart data={curves} margin={{ left: -10, right: 12 }}>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="epoch" {...axis} />
                    <YAxis {...axis} domain={k === "val" ? ["auto", 1] : [0, "auto"]} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} width={52} />
                    <Tooltip {...tooltipStyle} formatter={(v) => pct(Number(v), 2)} labelFormatter={(v) => `epoch ${v}`} />
                    <Legend wrapperStyle={{ fontSize: 11 }} iconType="plainline" />
                    {m.models.map((r, i) => (
                      <Line key={r.run} dataKey={`${r.run}:${k}`} name={r.display_name} stroke={MODEL_COLORS[i % MODEL_COLORS.length]} strokeWidth={2} dot={{ r: 4 }} connectNulls={false} isAnimationActive={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <ConfusionMatrix m={m} />

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-4">
          <div><CardTitle>Per-class performance</CardTitle><CardDescription>Deployed model; external sets cover only the classes they contain</CardDescription></div>
          <Tabs value={pcSet} onValueChange={(v) => setPcSet(v as SetKey)}>
            <TabsList>{SETS.map((s) => <TabsTrigger key={s.key} value={s.key}>{s.label.split(" (")[0]}</TabsTrigger>)}</TabsList>
          </Tabs>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table data-testid="per-class-table">
            <TableHeader><TableRow><TableHead>Class</TableHead><TableHead className="text-right">Precision</TableHead><TableHead className="text-right">Recall</TableHead><TableHead className="w-[30%]">F1</TableHead><TableHead className="text-right">Support</TableHead></TableRow></TableHeader>
            <TableBody>
              {Object.entries(m.selected_detail.per_class[pcSet] ?? {}).sort((a, b) => a[1].f1 - b[1].f1).map(([cls, v]) => {
                const pc = prettyClass(cls);
                return (
                  <TableRow key={cls}>
                    <TableCell><span className="font-medium">{pc.crop}</span> <span className="text-muted-foreground">— {pc.disease}</span></TableCell>
                    <TableCell className="text-right tabular-nums">{pct(v.precision)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(v.recall)}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="h-2 flex-1 rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${v.f1 * 100}%` }} /></div>
                        <span className="w-12 text-right text-xs tabular-nums">{pct(v.f1)}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">{v.support}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">Evaluation generated {new Date(m.generated_at).toLocaleString()} · model {m.serving.model_version}</p>
    </>
  );
}

function Kpi({ icon: Icon, label, value, note }: { icon: typeof Award; label: string; value: string; note: string }) {
  return (
    <Card className="gap-1 p-5">
      <div className="flex items-center gap-2 text-sm text-muted-foreground"><Icon className="size-4 text-primary" />{label}</div>
      <div className="font-heading text-3xl font-semibold tabular-nums">{value}</div>
      <div className="text-xs text-muted-foreground">{note}</div>
    </Card>
  );
}

function ConfusionMatrix({ m }: { m: MetricsSummary }) {
  const cm = m.selected_detail.confusion_matrix_test;
  const classes = m.selected_detail.classes;
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const norm = cm.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s); });
  const short = (c: string) => { const p = prettyClass(c); return `${p.crop}: ${p.disease}`; };
  return (
    <Card>
      <CardHeader>
        <CardTitle>Confusion matrix — PlantVillage test (row-normalised)</CardTitle>
        <CardDescription>Rows are true classes, columns predictions. Hover a cell for counts. Darker = larger share of the row.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 lg:grid-cols-[1fr_260px]">
          <div className="overflow-x-auto">
            <div className="grid min-w-[560px] gap-px rounded-md bg-border p-px" style={{ gridTemplateColumns: `repeat(${classes.length}, minmax(0, 1fr))` }}
              onMouseLeave={() => setHover(null)} data-testid="confusion-matrix">
              {norm.map((row, i) => row.map((v, j) => (
                <div key={`${i}-${j}`} onMouseEnter={() => setHover({ i, j })}
                  className={cn("aspect-square", hover && (hover.i === i || hover.j === j) && "outline outline-1 outline-foreground/30")}
                  style={{ background: v === 0 ? "var(--card)" : `color-mix(in oklch, var(--primary) ${Math.max(8, Math.sqrt(v) * 100)}%, var(--card))` }} />
              )))}
            </div>
          </div>
          <div className="space-y-3 text-sm">
            {hover ? (
              <div className="rounded-xl border border-border p-4">
                <div className="text-xs text-muted-foreground">True</div><div className="font-medium">{short(classes[hover.i])}</div>
                <div className="mt-2 text-xs text-muted-foreground">Predicted</div><div className="font-medium">{short(classes[hover.j])}</div>
                <div className="mt-2 font-heading text-2xl tabular-nums">{cm[hover.i][hover.j]} <span className="text-sm text-muted-foreground">({pct(norm[hover.i][hover.j])} of row)</span></div>
              </div>
            ) : <div className="rounded-xl border border-dashed border-border p-4 text-muted-foreground">Hover a cell</div>}
            <div>
              <div className="mb-1 font-semibold">Most frequent confusions</div>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {(m.selected_detail.top_confusions.test ?? []).slice(0, 6).map((c) => (
                  <li key={`${c.true}${c.pred}`}>{short(c.true)} → {prettyClass(c.pred).disease} <span className="tabular-nums">({c.count})</span></li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
