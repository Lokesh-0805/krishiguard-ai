"use client";

import type { ReactNode } from "react";
import { AlertTriangle, ExternalLink, FlaskConical, Leaf, ListChecks, ShieldCheck, Sprout, Bug } from "lucide-react";
import { TabsContent } from "@/components/ui/tabs";
import type { Advisory } from "@/lib/api";
import { useLang } from "@/lib/i18n";

function List({ items }: { items: string[] | null | undefined }) {
  if (!items?.length) return null;
  return (
    <ul className="space-y-2">
      {items.map((x, i) => (
        <li key={i} className="flex gap-2.5 text-sm leading-relaxed"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary" />{x}</li>
      ))}
    </ul>
  );
}

function Section({ icon: Icon, title, children }: { icon: typeof Leaf; title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="flex items-center gap-2 text-base font-semibold"><Icon className="size-4 text-primary" />{title}</h3>
      {children}
    </section>
  );
}

export function AdvisoryBody({ a, mode }: { a: Advisory; mode: "tabs" | "sections" }) {
  const { t } = useLang();
  const overview = (
    <div className="space-y-4">
      {a.summary && <p className="leading-relaxed">{a.summary}</p>}
      {a.causes?.length ? <Section icon={Bug} title={t("causes")}><List items={a.causes} /></Section> : null}
      {a.disclaimer && <p className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">{a.disclaimer}</p>}
    </div>
  );
  const symptoms = <List items={a.symptoms} />;
  const treatment = (
    <div className="space-y-6">
      {a.organic_treatment?.length ? (
        <Section icon={Sprout} title={t("organic")}><List items={a.organic_treatment} /></Section>
      ) : null}
      <Section icon={FlaskConical} title={t("chemical")}>
        {a.chemical_treatment?.items?.length ? (
          <>
            <List items={a.chemical_treatment.items} />
            {a.chemical_treatment.active_ingredients.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {a.chemical_treatment.active_ingredients.map((x) => (
                  <span key={x} className="rounded-md border border-border bg-muted px-2 py-0.5 font-mono text-xs">{x}</span>
                ))}
              </div>
            )}
            {a.chemical_label_notice && (
              <div className="flex gap-2 rounded-lg border border-warning/50 bg-warning/10 p-3 text-xs">
                <AlertTriangle className="size-4 shrink-0 text-warning" />{a.chemical_label_notice}
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">{t("no_chemical")}</p>
        )}
      </Section>
    </div>
  );
  const prevention = <List items={a.prevention} />;
  const sources = (
    <ul className="space-y-2.5">
      {a.sources.map((s) => (
        <li key={s.url} className="text-sm">
          <a href={s.url} target="_blank" rel="noreferrer" className="inline-flex items-start gap-1.5 font-medium text-primary hover:underline">
            <ExternalLink className="mt-0.5 size-3.5 shrink-0" />{s.title}
          </a>
          <div className="pl-5 text-xs text-muted-foreground">{s.publisher}{s.accessed ? ` · accessed ${s.accessed}` : ""}</div>
        </li>
      ))}
    </ul>
  );

  if (mode === "tabs") {
    return (
      <>
        <TabsContent value="overview">{overview}</TabsContent>
        <TabsContent value="symptoms">{symptoms}</TabsContent>
        {!a.healthy && <TabsContent value="treatment">{treatment}</TabsContent>}
        <TabsContent value="prevention">{prevention}</TabsContent>
        <TabsContent value="sources">{sources}</TabsContent>
      </>
    );
  }
  return (
    <div className="space-y-7">
      {overview}
      <Section icon={ListChecks} title={t("symptoms")}>{symptoms}</Section>
      {!a.healthy && treatment}
      <Section icon={ShieldCheck} title={t("prevention")}>{prevention}</Section>
      <Section icon={ExternalLink} title={t("sources")}>{sources}</Section>
    </div>
  );
}
