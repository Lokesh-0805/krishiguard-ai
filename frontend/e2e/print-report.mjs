// Prints docs/report/KrishiGuard_AI_Report.html to PDF with Chrome (correct rendering of all scripts, images, tables).
//   node e2e/print-report.mjs
import { chromium } from "@playwright/test";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(here, "../../docs/report");
const b = await chromium.launch({ channel: "chrome" });
const page = await b.newPage();
await page.goto(pathToFileURL(path.join(dir, "KrishiGuard_AI_Report.html")).href, { waitUntil: "load" });
await page.pdf({
  path: path.join(dir, "KrishiGuard_AI_Report.pdf"),
  format: "A4",
  printBackground: true,
  preferCSSPageSize: true,
  displayHeaderFooter: true,
  headerTemplate: "<div></div>",
  footerTemplate: `<div style="font-size:8px;color:#6b776f;width:100%;padding:0 14mm;display:flex;justify-content:space-between;font-family:Helvetica,Arial,sans-serif">
    <span>KrishiGuard AI — Crop Disease Detection and Advisory System</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`,
});
await b.close();
console.log("PDF written to", path.join(dir, "KrishiGuard_AI_Report.pdf"));
