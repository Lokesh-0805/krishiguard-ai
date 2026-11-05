"use client";

import Link from "next/link";
import {
  ArrowRight, BadgeCheck, BookOpenCheck, Camera, Eye, Gauge, Languages, Leaf, Lock, ScanSearch, ShieldQuestion, Sprout,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { LeafScan } from "@/components/site/leaf-scan";
import { useLang } from "@/lib/i18n";
import { pct, useDiseases, useMetrics } from "@/lib/hooks";

const COPY = {
  en: {
    pills: ["38 conditions · 14 crops", "English + हिंदी", "Explains its reasoning", "Says when it is unsure"],
    stats_title: "Measured, not marketed",
    stats_sub: "Numbers below are loaded live from the evaluation of the deployed model. Lab photos and real field photos are reported separately because they behave very differently.",
    lab: "Lab-style photos", lab_note: "PlantVillage held-out test set",
    field: "Real field photos", field_note: "PlantDoc test set (never trained on)",
    tomato: "Tomato field photos", tomato_note: "Independent dataset from Pakistan",
    f1: "macro-F1",
    how_title: "Three steps to a second opinion",
    steps: [
      ["Capture", "Photograph a single affected leaf in good light, or upload an existing photo."],
      ["Analyse", "A fine-tuned deep-learning model identifies crop and condition, calibrates its confidence and flags blurry photos."],
      ["Act", "Read symptoms, prevention and organic/chemical options from cited agricultural sources — in English or Hindi."],
    ],
    feat_title: "Built for trust in the field",
    feats: [
      ["Calibrated confidence", "Temperature-scaled probabilities with a validation-selected threshold; uncertain results are labelled as such."],
      ["Grad-CAM explanations", "See which leaf regions influenced the decision — a sanity check, not a guarantee."],
      ["Bilingual advisory", "Simple Hindi and English guidance, with sources and label-safety notices for any chemical."],
      ["Privacy by default", "Photos are analysed in memory and never stored; only anonymous prediction metadata is logged."],
      ["Works on any device", "Responsive web app with browser camera capture — no app install needed."],
      ["Honest evaluation", "Leakage-free splits, untouched test sets and external field data, with per-class results published."],
    ],
    crops_title: "Supported crops",
    cta_title: "Have a sick-looking leaf?",
    cta_sub: "Get a calibrated, explained diagnosis in seconds.",
  },
  hi: {
    pills: ["38 स्थितियाँ · 14 फसलें", "हिंदी + English", "कारण भी दिखाता है", "अनिश्चित होने पर बताता है"],
    stats_title: "दावे नहीं, मापे गए नतीजे",
    stats_sub: "नीचे के आँकड़े चल रहे मॉडल की जाँच से सीधे लोड होते हैं। लैब जैसी फ़ोटो और असली खेत की फ़ोटो के नतीजे अलग दिखाए गए हैं, क्योंकि दोनों में बड़ा अंतर होता है।",
    lab: "लैब जैसी फ़ोटो", lab_note: "PlantVillage का अलग रखा गया टेस्ट सेट",
    field: "असली खेत की फ़ोटो", field_note: "PlantDoc टेस्ट सेट (प्रशिक्षण में कभी नहीं)",
    tomato: "टमाटर खेत की फ़ोटो", tomato_note: "पाकिस्तान का स्वतंत्र डेटासेट",
    f1: "macro-F1",
    how_title: "तीन आसान कदम",
    steps: [
      ["फ़ोटो लें", "अच्छी रोशनी में रोग वाली एक पत्ती की फ़ोटो लें या पहले से ली गई फ़ोटो अपलोड करें।"],
      ["जाँच", "प्रशिक्षित डीप-लर्निंग मॉडल फसल और रोग पहचानता है, अपना भरोसा मापता है और धुंधली फ़ोटो पर चेतावनी देता है।"],
      ["कदम उठाएँ", "लक्षण, बचाव और जैविक/रासायनिक उपाय पढ़ें — प्रमाणित कृषि स्रोतों के साथ, हिंदी या अंग्रेज़ी में।"],
    ],
    feat_title: "खेत में भरोसे के लिए बनाया गया",
    feats: [
      ["मापा हुआ भरोसा", "कैलिब्रेटेड संभावना और जाँच-डेटा से चुनी गई सीमा; अनिश्चित नतीजे साफ़ बताए जाते हैं।"],
      ["Grad-CAM व्याख्या", "देखें कि पत्ती के किस हिस्से ने फ़ैसले पर असर डाला — यह जाँच में मदद है, गारंटी नहीं।"],
      ["दो भाषाओं में सलाह", "सरल हिंदी और अंग्रेज़ी में सलाह, स्रोतों और रसायनों के लिए लेबल-सुरक्षा सूचना के साथ।"],
      ["निजता सबसे पहले", "फ़ोटो सिर्फ़ जाँच के लिए उपयोग होती है, सहेजी नहीं जाती।"],
      ["हर डिवाइस पर", "मोबाइल ब्राउज़र में कैमरे से फ़ोटो — कोई ऐप इंस्टॉल नहीं।"],
      ["ईमानदार मूल्यांकन", "बिना डेटा-लीक के बँटवारा, अलग टेस्ट सेट और असली खेत के डेटा पर जाँच।"],
    ],
    crops_title: "समर्थित फसलें",
    cta_title: "पत्ती बीमार लग रही है?",
    cta_sub: "कुछ ही सेकंड में भरोसे के स्तर और व्याख्या के साथ जाँच पाएँ।",
  },
} as const;

const FEAT_ICONS = [Gauge, Eye, Languages, Lock, Camera, BadgeCheck];
const STEP_ICONS = [Camera, ScanSearch, Sprout];

export default function Home() {
  const { t, lang } = useLang();
  const c = COPY[lang];
  const metrics = useMetrics();
  const diseases = useDiseases(lang);
  const sel = metrics.data?.models.find((m) => m.selected);
  const crops = diseases.data ? Array.from(new Set(diseases.data.map((d) => d.crop))) : [];

  return (
    <>
      <section className="relative overflow-hidden bg-field">
        <div className="grid-lines pointer-events-none absolute inset-0 opacity-60" />
        <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 pb-20 pt-14 sm:px-6 lg:grid-cols-[1.1fr_0.9fr] lg:pt-20">
          <div className="space-y-7">
            <Badge variant="secondary" className="gap-1.5 rounded-full px-3 py-1 text-xs">
              <Leaf className="size-3.5" /> {t("hero_kicker")}
            </Badge>
            <h1 className="text-4xl font-semibold leading-[1.08] sm:text-5xl lg:text-6xl">{t("hero_title")}</h1>
            <p className="max-w-xl text-lg leading-relaxed text-muted-foreground">{t("hero_sub")}</p>
            <div className="flex flex-wrap gap-3">
              <Button asChild size="xl" data-testid="hero-cta">
                <Link href="/detect"><ScanSearch /> {t("cta_diagnose")}</Link>
              </Button>
              <Button asChild size="xl" variant="outline">
                <Link href="/performance">{t("hero_secondary")} <ArrowRight /></Link>
              </Button>
            </div>
            <ul className="flex flex-wrap gap-2 pt-1">
              {c.pills.map((p) => (
                <li key={p} className="rounded-full border border-border bg-background/70 px-3 py-1 text-xs font-medium text-muted-foreground">{p}</li>
              ))}
            </ul>
          </div>
          <LeafScan />
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
        <div className="mb-8 max-w-2xl space-y-2">
          <h2 className="text-3xl font-semibold">{c.stats_title}</h2>
          <p className="text-muted-foreground">{c.stats_sub}</p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {([
            ["test", c.lab, c.lab_note],
            ["external_test_plantdoc", c.field, c.field_note],
            ["external_test_tomato_pk_field", c.tomato, c.tomato_note],
          ] as const).map(([k, title, note], i) => {
            const r = sel?.results[k];
            return (
              <div key={k} className="rounded-2xl border border-border bg-card p-6 shadow-xs">
                <div className="text-sm font-medium text-muted-foreground">{title}</div>
                {metrics.loading ? (
                  <Skeleton className="mt-3 h-10 w-32" />
                ) : r ? (
                  <div className="mt-2 flex items-baseline gap-3">
                    <span className={`font-heading text-4xl font-semibold ${i === 0 ? "text-primary" : ""}`}>{pct(r.accuracy)}</span>
                    <span className="text-sm text-muted-foreground">acc · {pct(r.macro_f1)} {c.f1}</span>
                  </div>
                ) : (
                  <div className="mt-3 text-sm text-muted-foreground">{t("backend_down")}</div>
                )}
                <div className="mt-2 text-xs text-muted-foreground">{note}{r ? ` · n=${r.n.toLocaleString()}` : ""}</div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="border-y border-border/60 bg-muted/30">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
          <h2 className="mb-10 text-3xl font-semibold">{c.how_title}</h2>
          <ol className="grid gap-6 md:grid-cols-3">
            {c.steps.map(([title, body], i) => {
              const Icon = STEP_ICONS[i];
              return (
                <li key={title} className="relative rounded-2xl border border-border bg-card p-6">
                  <span className="absolute right-5 top-4 font-heading text-5xl font-semibold text-muted-foreground/15">{i + 1}</span>
                  <div className="mb-4 inline-flex size-11 items-center justify-center rounded-xl bg-secondary text-secondary-foreground"><Icon className="size-5" /></div>
                  <h3 className="mb-1.5 text-xl font-semibold">{title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">{body}</p>
                </li>
              );
            })}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
        <h2 className="mb-10 text-3xl font-semibold">{c.feat_title}</h2>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {c.feats.map(([title, body], i) => {
            const Icon = FEAT_ICONS[i];
            return (
              <div key={title} className="group rounded-2xl border border-border bg-card p-6 transition-shadow hover:shadow-md">
                <Icon className="mb-4 size-6 text-primary transition-transform group-hover:-translate-y-0.5" />
                <h3 className="mb-1.5 text-lg font-semibold">{title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{body}</p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="rounded-3xl border border-border bg-card p-8">
          <div className="mb-5 flex items-center gap-2">
            <BookOpenCheck className="size-5 text-primary" />
            <h2 className="text-2xl font-semibold">{c.crops_title}</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {diseases.loading
              ? Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="h-8 w-24 rounded-full" />)
              : crops.map((cr) => (
                  <Link key={cr} href={`/diseases#crop=${encodeURIComponent(cr)}`} className="rounded-full border border-border bg-secondary/60 px-4 py-1.5 text-sm font-medium hover:bg-secondary">
                    {cr}
                  </Link>
                ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pt-16 sm:px-6">
        <div className="relative overflow-hidden rounded-3xl bg-primary px-8 py-12 text-primary-foreground sm:px-12">
          <ShieldQuestion className="absolute -right-6 -top-6 size-48 opacity-10" />
          <h2 className="text-3xl font-semibold">{c.cta_title}</h2>
          <p className="mt-2 max-w-xl opacity-90">{c.cta_sub}</p>
          <Button asChild size="xl" variant="secondary" className="mt-6">
            <Link href="/detect"><ScanSearch /> {t("cta_diagnose")}</Link>
          </Button>
        </div>
      </section>
    </>
  );
}
