"use client";

import { useEffect, useMemo, useState } from "react";
import { Bug, ChevronRight, Leaf, Search, ShieldCheck, Virus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { AdvisoryBody } from "@/components/advisory/advisory-body";
import { getAdvisory, type Advisory, type DiseaseSummary } from "@/lib/api";
import { useDiseases } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";
import { cn } from "@/lib/utils";

const PATHOGEN_ICON: Record<string, typeof Bug> = { virus: Virus, bacterium: Bug, mite: Bug };

function parseHash(): Record<string, string> {
  return Object.fromEntries(new URLSearchParams(window.location.hash.slice(1)));
}

export default function DiseasesPage() {
  const { t, lang } = useLang();
  const { data, loading, error } = useDiseases(lang);
  const [q, setQ] = useState("");
  const [crop, setCrop] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [adv, setAdv] = useState<Advisory | null>(null);

  // deep links: /diseases#id=<slug> or #crop=<name>
  useEffect(() => {
    const apply = () => {
      const h = parseHash();
      if (h.id) setOpenId(h.id);
      if (h.crop) setCrop(h.crop);
    };
    apply();
    window.addEventListener("hashchange", apply);
    return () => window.removeEventListener("hashchange", apply);
  }, []);

  useEffect(() => {
    if (!openId) return;
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset while the new advisory loads
    setAdv(null);
    getAdvisory(openId, lang).then((a) => alive && setAdv(a)).catch(() => alive && setAdv(null));
    return () => { alive = false; };
  }, [openId, lang]);

  const crops = useMemo(() => Array.from(new Set((data ?? []).map((d) => d.crop))), [data]);
  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (data ?? []).filter((d) =>
      (!crop || d.crop === crop) &&
      (!s || `${d.crop} ${d.name} ${d.scientific_name ?? ""}`.toLowerCase().includes(s)));
  }, [data, q, crop]);
  const grouped = useMemo(() => {
    const m = new Map<string, DiseaseSummary[]>();
    filtered.forEach((d) => m.set(d.crop, [...(m.get(d.crop) ?? []), d]));
    return [...m.entries()];
  }, [filtered]);

  const close = () => {
    setOpenId(null);
    history.replaceState(null, "", window.location.pathname);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="mb-8 max-w-2xl space-y-2">
        <h1 className="text-3xl font-semibold sm:text-4xl">{t("library_title")}</h1>
        <p className="text-muted-foreground">{t("library_sub")}</p>
      </div>
      <div className="mb-6 space-y-3">
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("search")}
            className="h-11 w-full rounded-xl border border-input bg-card pl-10 pr-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/40" data-testid="library-search" />
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Chip active={!crop} onClick={() => setCrop(null)}>{t("all_crops")}</Chip>
          {crops.map((c) => <Chip key={c} active={crop === c} onClick={() => setCrop(crop === c ? null : c)}>{c}</Chip>)}
        </div>
      </div>

      {error && <p className="text-destructive">{t("backend_down")}</p>}
      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{Array.from({ length: 9 }).map((_, i) => <Skeleton key={i} className="h-28 rounded-2xl" />)}</div>
      ) : (
        <div className="space-y-10">
          {grouped.map(([cropName, items]) => (
            <section key={cropName}>
              <h2 className="mb-3 flex items-center gap-2 text-xl font-semibold"><Leaf className="size-5 text-primary" />{cropName}
                <span className="text-sm font-normal text-muted-foreground">({items.length})</span></h2>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {items.map((d) => {
                  const Icon = d.healthy ? ShieldCheck : PATHOGEN_ICON[d.pathogen_type ?? ""] ?? Bug;
                  return (
                    <button key={d.id} onClick={() => setOpenId(d.slug)} className="text-left" data-testid="disease-card">
                      <Card className="group flex h-full flex-row items-center gap-4 p-4 transition-all hover:-translate-y-0.5 hover:shadow-md">
                        <div className={cn("flex size-11 shrink-0 items-center justify-center rounded-xl", d.healthy ? "bg-success/15 text-success" : "bg-accent text-accent-foreground")}>
                          <Icon className="size-5" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="font-semibold">{d.name}</div>
                          <div className="truncate text-xs italic text-muted-foreground">{d.scientific_name ?? (d.healthy ? t("healthy") : "")}</div>
                          {d.pathogen_type && d.pathogen_type !== "none" && <Badge variant="secondary" className="mt-1.5 text-[10px] capitalize">{d.pathogen_type}</Badge>}
                        </div>
                        <ChevronRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                      </Card>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}

      <Sheet open={!!openId} onOpenChange={(o) => !o && close()}>
        <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
          <SheetHeader>
            <SheetTitle className="font-heading text-2xl">{adv ? adv.name : t("loading")}</SheetTitle>
            <SheetDescription>{adv ? `${adv.crop}${adv.scientific_name ? ` · ${adv.scientific_name}` : ""}` : ""}</SheetDescription>
          </SheetHeader>
          <div className="px-4 pb-8">{adv ? <AdvisoryBody a={adv} mode="sections" /> : <Skeleton className="h-64 w-full" />}</div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} className={cn("rounded-full border px-3 py-1 text-sm font-medium transition-colors",
      active ? "border-primary bg-primary text-primary-foreground" : "border-border bg-card hover:bg-muted")}>{children}</button>
  );
}
