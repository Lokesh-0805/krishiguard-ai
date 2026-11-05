import type { Metadata, Viewport } from "next";
import { Fraunces, Inter, Noto_Sans_Devanagari } from "next/font/google";
import { SiteHeader } from "@/components/site/header";
import { SiteFooter } from "@/components/site/footer";
import { LanguageProvider } from "@/lib/i18n";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const fraunces = Fraunces({ variable: "--font-display", subsets: ["latin"], axes: ["opsz"] });
const deva = Noto_Sans_Devanagari({ variable: "--font-deva", subsets: ["devanagari"], weight: ["400", "500", "600", "700"] });

export const metadata: Metadata = {
  title: { default: "KrishiGuard AI — Crop Disease Detection & Advisory", template: "%s · KrishiGuard AI" },
  description:
    "Upload a leaf photo to identify crop diseases with calibrated confidence, Grad-CAM explanations and bilingual (English/Hindi) advisory from agricultural sources.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8f7f0" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1a14" },
  ],
};

const themeScript = `try{var t=localStorage.getItem('kg-theme');if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme: dark)').matches))document.documentElement.classList.add('dark')}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${fraunces.variable} ${deva.variable} h-full antialiased`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="flex min-h-full flex-col">
        <LanguageProvider>
          <TooltipProvider delayDuration={150}>
            <SiteHeader />
            <main className="flex-1">{children}</main>
            <SiteFooter />
          </TooltipProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
