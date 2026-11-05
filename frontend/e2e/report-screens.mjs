// Captures the UI screenshots used in docs/report (run with frontend :3000 and backend :8000 up).
//   node e2e/report-screens.mjs
import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "../..");
const OUT = path.join(ROOT, "docs/report/screens");
const FIX = JSON.parse(fs.readFileSync(path.join(here, "fixtures.json"), "utf8"));
const leaf = path.join(ROOT, "ml/datasets/raw", FIX.confident_tomato_early_blight);
const field = path.join(ROOT, "ml/datasets/raw", FIX.unrecognized_field_image);
const BASE = process.env.BASE ?? "http://localhost:3000";
fs.mkdirSync(OUT, { recursive: true });

const b = await chromium.launch({ channel: "chrome" });
const shot = async (p, name, full = false) => p.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: full });

// desktop
const d = await b.newPage({ viewport: { width: 1366, height: 860 }, deviceScaleFactor: 1.5 });
await d.goto(BASE + "/"); await d.waitForTimeout(1500); await shot(d, "landing");
await d.goto(BASE + "/detect"); await d.waitForTimeout(800); await shot(d, "detect-empty");
await d.getByTestId("file-input").setInputFiles(leaf); await d.waitForTimeout(600); await shot(d, "detect-preview");
await d.getByTestId("analyse").click(); await d.waitForURL(/results/); await d.waitForTimeout(1500);
await shot(d, "results-en");
await d.getByTestId("gradcam").scrollIntoViewIfNeeded();
await d.getByRole("button", { name: /Heatmap/ }).click(); await d.waitForTimeout(400);
await d.getByTestId("gradcam").screenshot({ path: path.join(OUT, "gradcam-viewer.png") });
await d.getByRole("tab", { name: /Treatment/ }).click(); await d.waitForTimeout(300);
await d.getByTestId("advisory").screenshot({ path: path.join(OUT, "advisory-treatment-en.png") });
await d.evaluate(() => window.scrollTo(0, 0));
await d.getByTestId("lang-toggle").click(); await d.waitForTimeout(1500);
await shot(d, "results-hi");
await d.getByRole("tab", { name: /लक्षण/ }).click(); await d.waitForTimeout(300);
await d.getByTestId("advisory").screenshot({ path: path.join(OUT, "advisory-symptoms-hi.png") });
await d.getByTestId("lang-toggle").click(); await d.waitForTimeout(800);
await d.goto(BASE + "/detect"); await d.getByTestId("file-input").setInputFiles(field);
await d.getByTestId("analyse").click(); await d.waitForURL(/results/); await d.waitForTimeout(1200);
await shot(d, "results-unrecognized");
await d.goto(BASE + "/diseases"); await d.waitForTimeout(1200); await shot(d, "library");
await d.getByTestId("library-search").fill("late blight"); await d.getByTestId("disease-card").first().click();
await d.waitForTimeout(1200); await shot(d, "library-detail");
await d.goto(BASE + "/performance"); await d.waitForTimeout(2000); await shot(d, "performance", true);
await d.goto(BASE + "/methodology"); await d.waitForTimeout(1500); await shot(d, "methodology");

// mobile
const m = await b.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
await m.goto(BASE + "/"); await m.waitForTimeout(1200); await shot(m, "mobile-landing");
await m.goto(BASE + "/detect"); await m.waitForTimeout(800); await shot(m, "mobile-detect");
await m.getByTestId("file-input").setInputFiles(leaf); await m.getByTestId("analyse").click();
await m.waitForURL(/results/); await m.waitForTimeout(1500); await shot(m, "mobile-results");
await b.close();
console.log("screens written to", OUT);
