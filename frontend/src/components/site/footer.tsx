"use client";

import Link from "next/link";
import { ShieldAlert } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { Logo } from "./logo";

export function SiteFooter() {
  const { t } = useLang();
  return (
    <footer className="mt-24 border-t border-border/60 bg-muted/40">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-10 sm:px-6 md:grid-cols-[1.4fr_1fr]">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Logo className="size-7" />
            <span className="font-heading font-semibold">KrishiGuard AI</span>
          </div>
          <p className="flex max-w-xl gap-2 text-sm text-muted-foreground">
            <ShieldAlert className="mt-0.5 size-4 shrink-0 text-warning" />
            {t("footer_disclaimer")}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Link className="text-muted-foreground hover:text-foreground" href="/detect">{t("nav_detect")}</Link>
          <Link className="text-muted-foreground hover:text-foreground" href="/diseases">{t("nav_library")}</Link>
          <Link className="text-muted-foreground hover:text-foreground" href="/performance">{t("nav_performance")}</Link>
          <Link className="text-muted-foreground hover:text-foreground" href="/methodology">{t("nav_method")}</Link>
          <Link className="text-muted-foreground hover:text-foreground" href="/about">{t("nav_about")}</Link>
          <a className="text-muted-foreground hover:text-foreground" href="/docs" target="_blank" rel="noreferrer">API docs</a>
        </div>
      </div>
      <div className="border-t border-border/60 py-4 text-center text-xs text-muted-foreground">
        Datasets: PlantVillage (CC BY-SA 3.0), PlantDoc (CC BY 4.0), Tomato field images, Pakistan (CC BY 4.0).
      </div>
    </footer>
  );
}
