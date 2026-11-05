"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Languages, Menu, Moon, Sun, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLang, type TKey } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { Logo } from "./logo";

const NAV: { href: string; key: TKey }[] = [
  { href: "/detect", key: "nav_detect" },
  { href: "/diseases", key: "nav_library" },
  { href: "/performance", key: "nav_performance" },
  { href: "/methodology", key: "nav_method" },
  { href: "/about", key: "nav_about" },
];

export function SiteHeader() {
  const { t, lang, setLang } = useLang();
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const toggleTheme = () => {
    const dark = document.documentElement.classList.toggle("dark");
    try { localStorage.setItem("kg-theme", dark ? "dark" : "light"); } catch { /* ignore */ }
  };
  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5" onClick={() => setOpen(false)}>
          <Logo />
          <span className="font-heading text-lg font-semibold tracking-tight">
            KrishiGuard <span className="text-primary">AI</span>
          </span>
        </Link>
        <nav className="ml-6 hidden items-center gap-1 lg:flex" aria-label="Main">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={cn(
                "rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                path === n.href && "bg-secondary text-secondary-foreground",
              )}
            >
              {t(n.key)}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setLang(lang === "en" ? "hi" : "en")}
            aria-label="Switch language / भाषा बदलें"
            data-testid="lang-toggle"
          >
            <Languages /> {t("lang_label")}
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={toggleTheme} aria-label="Toggle dark mode">
            <Sun className="hidden dark:block" />
            <Moon className="dark:hidden" />
          </Button>
          <Button asChild size="sm" className="hidden sm:inline-flex">
            <Link href="/detect">{t("cta_diagnose")}</Link>
          </Button>
          <Button variant="ghost" size="icon-sm" className="lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu" aria-expanded={open}>
            {open ? <X /> : <Menu />}
          </Button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-border/60 bg-background px-4 py-3 lg:hidden" aria-label="Mobile">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              onClick={() => setOpen(false)}
              className={cn("block rounded-lg px-3 py-2.5 text-sm font-medium", path === n.href ? "bg-secondary" : "hover:bg-muted")}
            >
              {t(n.key)}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
