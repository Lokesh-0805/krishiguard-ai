// Typed client for the KrishiGuard FastAPI backend. Requests go to same-origin /api/* which Next.js
// rewrites to the backend (see next.config.ts), so no CORS or secrets are needed in the browser.

export type Lang = "en" | "hi";
export type Status = "confident" | "uncertain" | "unrecognized";

export interface TopK { class_id: string; crop: string; disease: string; probability: number }
export interface ChemicalTreatment { items: string[]; active_ingredients: string[] }
export interface Source { title: string; publisher: string; url: string; accessed?: string }
export interface Advisory {
  id: string; slug: string; crop: string; name: string; healthy: boolean;
  pathogen_type: string | null; scientific_name: string | null;
  summary: string | null; symptoms: string[] | null; causes: string[] | null; prevention: string[] | null;
  organic_treatment: string[] | null; chemical_treatment: ChemicalTreatment | null;
  sources: Source[]; disclaimer: string | null; chemical_label_notice: string | null; language: Lang;
}
export interface Explanation {
  method: string; target_layer: string; model_input_png: string; heatmap_png: string; overlay_png: string; note: string;
}
export interface Prediction {
  prediction_id: string | null; model_version: string; status: Status; status_message: string;
  class_id: string; crop: string; disease: string; healthy: boolean; confidence: number; crop_confidence: number;
  top_k: TopK[]; quality: { width: number; height: number; blur_score: number; flags: string[] };
  latency_ms: number; advisory: Advisory | null; explanation: Explanation | null; filename: string | null;
}
export interface DiseaseSummary {
  id: string; slug: string; crop: string; name: string; healthy: boolean; pathogen_type: string | null; scientific_name: string | null;
}
export interface ApiErrorBody { error: { code: string; message: string; details?: Record<string, unknown> } }

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>;
  let code = "http_error", message = `Request failed (${res.status})`;
  try {
    const b = (await res.json()) as ApiErrorBody;
    code = b.error?.code ?? code; message = b.error?.message ?? message;
  } catch { /* non-JSON error (e.g. backend down behind the proxy) */ }
  if (res.status === 500 && code === "http_error") message = "The analysis service is unreachable. Please try again shortly.";
  throw new ApiError(res.status, code, message);
}

export async function predict(file: Blob, filename: string, lang: Lang, explain = true): Promise<Prediction> {
  const fd = new FormData();
  fd.append("file", file, filename);
  fd.append("lang", lang);
  fd.append("explain", String(explain));
  return handle<Prediction>(await fetch("/api/predict", { method: "POST", body: fd }));
}

export const getJSON = async <T,>(path: string): Promise<T> => handle<T>(await fetch(path, { cache: "no-store" }));
export const listDiseases = (lang: Lang) => getJSON<{ count: number; items: DiseaseSummary[] }>(`/api/diseases?lang=${lang}`);
export const getAdvisory = (id: string, lang: Lang) => getJSON<Advisory>(`/api/advisory/${encodeURIComponent(id)}?lang=${lang}`);
export const getMetrics = () => getJSON<MetricsSummary>("/api/metrics");
export const getHealth = () => getJSON<{ status: string; model_loaded: boolean; model_version: string | null }>("/api/health");

export async function sendFeedback(body: {
  prediction_id: string; prediction_correct?: boolean | null; suggested_class?: string | null; advisory_helpful?: boolean | null; comment?: string | null;
}) {
  return handle<{ id: number; verification_status: string }>(
    await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  );
}

// ---- metrics summary (written by ml/evaluation/compare_and_export.py) ----
export interface SetMetrics {
  n: number; accuracy: number; macro_precision: number; macro_recall: number; macro_f1: number; weighted_f1: number;
  crop_accuracy: number; top3_accuracy: number; ece_uncalibrated: number; ece_calibrated: number;
  selective_at_threshold: { threshold: number; coverage: number; selective_accuracy: number | null };
  unrecognized_rate: number;
}
export interface HistoryRow { epoch: number; train_loss: number; val_loss: number; val_accuracy: number; val_macro_f1: number; extval_macro_f1: number; train_acc: number }
export interface ModelRow {
  run: string; arch: string; display_name: string; img_size: number; trained_on: string; parameters_m: number; checkpoint_mb: number;
  epochs_trained: number; best_epoch: number; train_minutes: number; temperature: number; confidence_threshold: number;
  latency: Record<string, { batch1_ms_mean: number; batch1_ms_p50: number; batch1_ms_p95: number }>;
  results: Record<string, SetMetrics>; history: HistoryRow[]; selected: boolean;
}
export interface PerClass { precision: number; recall: number; f1: number; support: number }
export interface MetricsSummary {
  generated_at: string;
  selection: { rule: string; selected_run: string; validation_scores: Record<string, number> };
  serving: { model_version: string; arch: string; img_size: number; temperature: number; confidence_threshold: number; reject_threshold: number; size_mb: number; parameters_m: number };
  models: ModelRow[];
  selected_detail: {
    per_class: Record<string, Record<string, PerClass>>;
    confusion_matrix_test: number[][];
    top_confusions: Record<string, { true: string; pred: string; count: number }[]>;
    risk_coverage: Record<string, { coverage: number[]; selective_accuracy: number[]; aurc: number }>;
    classes: string[];
  };
  dataset: { splits: Record<string, number>; plantvillage_images: number; num_classes: number; num_crops: number; leakage_groups: number; external_test_by_source: Record<string, number> };
}

export function prettyClass(id: string): { crop: string; disease: string } {
  const [c, d = ""] = id.split("___");
  const crop = c.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", " (bell)").replace(/_/g, " ");
  return { crop, disease: d.replace(/_/g, " ").trim() };
}
