import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(__dirname, "../..");
const SHOTS = path.join(ROOT, "docs/screenshots");

// Held-out images chosen from the deployed model's cached test-set predictions (see fixtures.json).
const FIX = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures.json"), "utf8"));
const img = (rel: string) => path.join(ROOT, "ml/datasets/raw", rel);

test("full diagnosis journey", async ({ page }, info) => {
  const shot = (name: string) => page.screenshot({ path: path.join(SHOTS, `${info.project.name}-${name}.png`), fullPage: true });
  fs.mkdirSync(SHOTS, { recursive: true });

  // 1. open the website
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByText("Lab-style photos")).toBeVisible();
  await shot("01-landing");

  // 2-3. upload a leaf image and submit
  await page.getByTestId("hero-cta").click();
  await expect(page).toHaveURL(/\/detect/);
  await page.getByTestId("file-input").setInputFiles(img(FIX.confident_tomato_early_blight));
  await expect(page.getByTestId("preview")).toBeVisible();
  await shot("02-detect-preview");
  await page.getByTestId("analyse").click();

  // 4. receive the prediction
  await expect(page).toHaveURL(/\/results/);
  await expect(page.getByTestId("diagnosis-card")).toBeVisible();
  await expect(page.getByTestId("disease-name")).toContainText(/early blight/i);
  await expect(page.getByTestId("status-banner")).toHaveAttribute("data-status", "confident");

  // 5. disease information
  await expect(page.getByTestId("advisory")).toBeVisible();
  await page.getByRole("tab", { name: /Symptoms/ }).click();
  await expect(page.getByTestId("advisory").locator("li").first()).toBeVisible();

  // 7. inspect Grad-CAM
  await expect(page.getByTestId("gradcam").locator("img").first()).toBeVisible();
  await shot("03-results-en");

  // 6. switch to Hindi: labels and advisory reload in Devanagari
  await page.getByTestId("lang-toggle").click();
  await expect(page.getByTestId("disease-name")).toHaveText(/[ऀ-ॿ]/);
  await expect(page.getByTestId("advisory")).toContainText(/[ऀ-ॿ]/);
  await shot("04-results-hi");

  // report download
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-report").click()]);
  expect(dl.suggestedFilename()).toMatch(/krishiguard-report-.*\.html/);

  // 8. model performance dashboard (real metrics)
  await page.goto("/performance");
  await expect(page.getByTestId("kpis")).toBeVisible();
  await expect(page.getByTestId("model-table").locator("tbody tr")).not.toHaveCount(0);
  await expect(page.getByTestId("confusion-matrix")).toBeVisible();
  await shot("05-performance");
});

test("disease library search and detail sheet", async ({ page }, info) => {
  await page.goto("/diseases");
  await expect(page.getByTestId("disease-card").first()).toBeVisible();
  await page.getByTestId("library-search").fill("late blight");
  await expect(page.getByTestId("disease-card")).toHaveCount(2);
  await page.getByTestId("disease-card").first().click();
  await expect(page.getByRole("dialog")).toContainText(/Phytophthora infestans/i);
  await page.screenshot({ path: path.join(SHOTS, `${info.project.name}-06-library.png`) });
});

test("field photo outside the training distribution is reported as not recognised", async ({ page }) => {
  await page.goto("/detect");
  await page.getByTestId("file-input").setInputFiles(img(FIX.unrecognized_field_image));
  await page.getByTestId("analyse").click();
  await expect(page).toHaveURL(/\/results/);
  await expect(page.getByTestId("status-banner")).toHaveAttribute("data-status", "unrecognized");
  await expect(page.getByTestId("advisory")).toHaveCount(0);
});

test("'New diagnosis' after a previous diagnosis starts fresh (regression)", async ({ page }) => {
  // first diagnosis + feedback
  await page.goto("/detect");
  await page.getByTestId("file-input").setInputFiles(img(FIX.confident_tomato_early_blight));
  await page.getByTestId("analyse").click();
  await expect(page).toHaveURL(/\/results/);
  await expect(page.getByTestId("status-banner")).toHaveAttribute("data-status", "confident");
  await page.getByRole("button", { name: /Diagnosis correct/ }).click();
  await page.getByRole("button", { name: /Send feedback/ }).click();
  await expect(page.getByTestId("feedback-done")).toBeVisible();

  // back to the form via the results page button: must be a clean form, not stuck on "Analysing…"
  await page.getByRole("link", { name: /New diagnosis/ }).click();
  await expect(page).toHaveURL(/\/detect/);
  await expect(page.getByTestId("dropzone")).toBeVisible();
  await expect(page.getByText("Analysing…")).toHaveCount(0);

  // second diagnosis shows the NEW result and a fresh feedback box
  await page.getByTestId("file-input").setInputFiles(img(FIX.unrecognized_field_image));
  await expect(page.getByTestId("analyse")).toBeEnabled();
  await page.getByTestId("analyse").click();
  await expect(page).toHaveURL(/\/results/);
  await expect(page.getByTestId("status-banner")).toHaveAttribute("data-status", "unrecognized");
  await expect(page.getByTestId("feedback-done")).toHaveCount(0);
});

test("rejects a non-image upload with a clear error", async ({ page }) => {
  await page.goto("/detect");
  await page.getByTestId("file-input").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  await expect(page.getByTestId("detect-error")).toBeVisible();
});
