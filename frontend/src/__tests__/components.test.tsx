import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { AdvisoryBody } from "@/components/advisory/advisory-body";
import { GradCamViewer } from "@/components/results/gradcam-viewer";
import { LanguageProvider, useLang } from "@/lib/i18n";
import { buildReportHtml } from "@/lib/report";
import { prettyClass, type Advisory, type Prediction } from "@/lib/api";

const advisory: Advisory = {
  id: "Tomato___Early_blight", slug: "tomato-early-blight", crop: "Tomato", name: "Early blight", healthy: false,
  pathogen_type: "fungus", scientific_name: "Alternaria solani", summary: "A fungal disease.",
  symptoms: ["Concentric rings"], causes: ["Warm humid weather"], prevention: ["Crop rotation"],
  organic_treatment: ["Remove infected leaves"], chemical_treatment: { items: ["Option reported by source"], active_ingredients: ["mancozeb"] },
  sources: [{ title: "Extension page", publisher: "University", url: "https://example.org/a" }],
  disclaimer: "Consult an expert.", chemical_label_notice: "Follow the label.", language: "en",
};

function LangSwitch() {
  const { setLang, t } = useLang();
  return <button onClick={() => setLang("hi")}>{t("symptoms")}</button>;
}

describe("prettyClass", () => {
  it("formats PlantVillage ids", () => {
    expect(prettyClass("Corn_(maize)___Common_rust_")).toEqual({ crop: "Corn", disease: "Common rust" });
    expect(prettyClass("Pepper,_bell___healthy")).toEqual({ crop: "Pepper (bell)", disease: "healthy" });
  });
});

describe("i18n", () => {
  it("switches UI strings to Hindi", () => {
    render(<LanguageProvider><LangSwitch /></LanguageProvider>);
    const b = screen.getByRole("button");
    expect(b).toHaveTextContent("Symptoms");
    fireEvent.click(b);
    expect(b).toHaveTextContent("लक्षण");
  });
});

describe("AdvisoryBody", () => {
  it("renders every advisory section with sources and label notice", () => {
    render(<LanguageProvider><AdvisoryBody a={advisory} mode="sections" /></LanguageProvider>);
    expect(screen.getByText("Concentric rings")).toBeInTheDocument();
    expect(screen.getByText("mancozeb")).toBeInTheDocument();
    expect(screen.getByText("Follow the label.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Extension page/ })).toHaveAttribute("href", "https://example.org/a");
  });
  it("states when no chemical treatment is recommended", () => {
    render(<LanguageProvider><AdvisoryBody a={{ ...advisory, chemical_treatment: null }} mode="sections" /></LanguageProvider>);
    expect(screen.getByText(/No chemical treatment is recommended/)).toBeInTheDocument();
  });
});

describe("GradCamViewer", () => {
  it("shows original/heatmap/overlay in grid mode", () => {
    const e = { method: "Grad-CAM", target_layer: "bn2", model_input_png: "AAA", heatmap_png: "BBB", overlay_png: "CCC", note: "Not proof." };
    render(<LanguageProvider><GradCamViewer e={e} labels={{ original: "Model input", heatmap: "Heatmap", overlay: "Overlay" }} /></LanguageProvider>);
    fireEvent.click(screen.getByRole("button", { name: /Heatmap/ }));
    expect(screen.getByAltText("Heatmap")).toHaveAttribute("src", "data:image/png;base64,BBB");
    expect(screen.getByText(/not proof that the diagnosis is correct/)).toBeInTheDocument();
  });
});

describe("buildReportHtml", () => {
  it("escapes user/model text and includes the disclaimer", () => {
    const p = {
      prediction_id: "x", model_version: "m@1", status: "confident", status_message: "", class_id: advisory.id, crop: "Tomato",
      disease: "<script>", healthy: false, confidence: 0.9, crop_confidence: 0.95,
      top_k: [{ class_id: advisory.id, crop: "Tomato", disease: "<b>x</b>", probability: 0.9 }],
      quality: { width: 1, height: 1, blur_score: 1, flags: [] }, latency_ms: 1, advisory: null, explanation: null, filename: null,
    } as Prediction;
    const html = buildReportHtml({ stored: { prediction: p, image: "data:,", createdAt: new Date().toISOString() }, advisory, lang: "en",
      names: { crop: "Tomato", disease: "<script>alert(1)</script>" }, statusTitle: "Confident", statusMsg: "ok" });
    expect(html).not.toContain("<script>alert(1)</script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("not a substitute for an agricultural expert");
    expect(html).toContain("Follow the label.");
  });
});
