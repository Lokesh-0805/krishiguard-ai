"use client";

import { Database, FlaskConical, GitBranch, Microscope, Scale, ScanEye, Split } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Figure } from "@/components/site/figure";
import { pct, useMetrics } from "@/lib/hooks";
import { useLang } from "@/lib/i18n";

function Step({ icon: Icon, n, title, children }: { icon: typeof Database; n: number; title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-4 md:grid-cols-[220px_1fr]">
      <div className="flex items-start gap-3 md:flex-col">
        <div className="flex size-11 items-center justify-center rounded-xl bg-secondary text-secondary-foreground"><Icon className="size-5" /></div>
        <div><div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Step {n}</div><h2 className="text-xl font-semibold">{title}</h2></div>
      </div>
      <div className="space-y-3 leading-relaxed text-muted-foreground [&_b]:text-foreground">{children}</div>
    </section>
  );
}

export default function MethodologyPage() {
  const { t } = useLang();
  const { data } = useMetrics();
  const d = data?.dataset;
  const sel = data?.models.find((m) => m.selected);
  return (
    <div className="mx-auto max-w-6xl space-y-14 px-4 py-10 sm:px-6 sm:py-14">
      <div className="max-w-3xl space-y-3">
        <h1 className="text-3xl font-semibold sm:text-4xl">{t("method_title")}</h1>
        <p className="text-lg text-muted-foreground">How KrishiGuard was built and evaluated — designed so that the reported numbers are reproducible and not inflated by leakage.</p>
      </div>

      <Step icon={Database} n={1} title="Data">
        <p><b>PlantVillage</b> (Mohanty et al., 2016; CC BY-SA 3.0): {d ? d.plantvillage_images.toLocaleString() : "54,305"} lab-style leaf photos, {d?.num_classes ?? 38} classes across {d?.num_crops ?? 14} crops, downloaded from the authors&apos; repository. Used for training, validation and internal testing.</p>
        <p><b>PlantDoc</b> (Singh et al., 2020; CC BY 4.0): real-world photos with cluttered backgrounds; all 28 classes were mapped onto PlantVillage labels. Its training release was split into an <i>external validation</i> part (calibration and model selection only) and a <i>domain-train</i> part (used only in a clearly labelled adaptation experiment); its test release is untouched.</p>
        <p><b>Tomato field images, Pakistan</b> (Mendeley Data, CC BY 4.0): 7,200 smartphone photos from fields, six tomato classes. Never used for training, tuning or selection — a fully independent external test.</p>
        <div className="grid gap-4 pt-2 sm:grid-cols-2"><Figure src="/figures/class_distribution_plantvillage.png" caption="Images per class (PlantVillage) — 36× imbalance between largest and smallest class" /><Figure src="/figures/samples_plantdoc.png" caption="PlantDoc samples — real backgrounds, lighting and multiple leaves" /></div>
      </Step>

      <Step icon={Split} n={2} title="Leakage-free splitting">
        <p>PlantVillage contains several photos of the same physical leaf. Random image-level splits put near-identical photos in train and test and inflate accuracy. We grouped images using the dataset&apos;s own <b>leaf-identity map</b>, exact byte duplicates (MD5) and <b>perceptual-hash near-duplicates</b> (64-bit pHash, Hamming ≤ 4), producing {d ? d.leakage_groups.toLocaleString() : "~20k"} groups. Whole groups were assigned to train / validation / test (≈70/15/15, stratified by class, seed 42).</p>
        <p>External images that near-duplicate any PlantVillage image, and PlantDoc test images that duplicate PlantDoc train images, were removed from the external sets.</p>
        {d && <p className="text-sm">Split sizes: {Object.entries(d.splits).map(([k, v]) => `${k.replace(/_/g, " ")} ${v.toLocaleString()}`).join(" · ")}</p>}
        <div className="grid gap-4 pt-2 sm:grid-cols-2"><Figure src="/figures/split_distribution.png" caption="Per-class group split" /><Figure src="/figures/rgb_distribution.png" caption="Colour statistics differ strongly between lab and field datasets (domain shift)" /></div>
      </Step>

      <Step icon={FlaskConical} n={3} title="Preprocessing & augmentation">
        <p>Training: random resized crop (scale 0.55–1), horizontal/vertical flips, rotation up to 30°, brightness/contrast/saturation jitter, very small hue jitter (yellowing and necrosis colour are diagnostic), occasional Gaussian blur to mimic phone cameras. Evaluation and serving: resize the short side to size/0.875 and centre-crop — the API uses a re-implementation that is unit-tested to be numerically identical.</p>
      </Step>

      <Step icon={GitBranch} n={4} title="Model & training">
        <p>An ImageNet-pretrained <b>EfficientNet-B0</b> (4.1 M parameters, 224 px input) was fine-tuned in PyTorch with transfer learning: new 38-class head, AdamW with a higher learning rate on the head than on the backbone, cosine schedule with warm-up, class-weighted label-smoothed cross-entropy (against the 36× class imbalance), early stopping on validation macro-F1 and MLflow tracking.</p>
        <p>Hyper-parameters were chosen with a small <b>Optuna</b> search (8 one-epoch trials on a 4,000-image subset). The final model trained for 5 epochs (~1h40 on an Apple M1 GPU) and the best epoch by validation macro-F1 was kept{sel ? ` (epoch ${sel.best_epoch})` : ""}.</p>
        <p className="text-sm">Scope note: additional architectures (ResNet-50, ViT, EfficientNet-B2) were planned but not trained, to keep the computational load suitable for a laptop; one short ResNet-50 learning-rate probe was run during tuning.</p>
      </Step>

      <Step icon={Scale} n={5} title="Calibration & uncertainty">
        <p>Softmax scores are over-confident and say nothing about correctness on unfamiliar photos. We fit a single <b>temperature</b> on the validation sets (lab and field weighted equally) and select a <b>confidence threshold</b> on validation data only. Below it the app shows the result as <i>uncertain</i>, lists alternatives and withholds treatment advice; far below it (or for blurry photos) the image is reported as not recognised.</p>
        {sel && <p className="text-sm">Deployed values: T = {data!.serving.temperature.toFixed(2)}, confident threshold {pct(data!.serving.confidence_threshold)}, reject threshold {pct(data!.serving.reject_threshold)}.</p>}
        <div className="grid gap-4 pt-2 sm:grid-cols-2"><Figure src="/figures/risk_coverage.png" caption="Risk–coverage curves of the deployed model" /><Figure src="/figures/reliability_external_test_plantdoc.png" caption="Reliability before/after temperature scaling (PlantDoc test)" /></div>
      </Step>

      <Step icon={ScanEye} n={6} title="Explainability">
        <p><b>Grad-CAM</b> is computed on the deployed network&apos;s last convolutional feature map (patch tokens for ViTs) for the predicted class, on exactly the cropped region the model saw. It shows which regions influenced the score; it is not evidence that the diagnosis is right.</p>
        <Figure src="/figures/gradcam_examples.png" caption="Grad-CAM on lab (top) and field (bottom) images — real outputs of the deployed model" />
      </Step>

      <Step icon={Microscope} n={7} title="Evaluation protocol">
        <p>Test and external-test sets are evaluated once with frozen temperature, thresholds and model choice. We report accuracy, macro and weighted precision/recall/F1, per-class metrics, confusion matrices, calibration error (ECE), selective accuracy and CPU latency. Macro averages on external sets are taken over the classes present in that set.</p>
        <div className="grid gap-4 pt-2 sm:grid-cols-2"><Figure src="/figures/model_comparison.png" caption="Internal (lab) vs external (field) macro-F1 of the trained model" /><Figure src="/figures/confusion_matrix_test_normalized.png" caption="Deployed model — PlantVillage test confusion matrix" /></div>
      </Step>

      <Card className="bg-muted/40"><CardContent className="text-sm text-muted-foreground">
        Full details: <code>docs/ML_METHODOLOGY.md</code>, <code>docs/DATASET_REPORT.md</code>, <code>docs/MODEL_EVALUATION_REPORT.md</code> and <code>docs/EXPERIMENT_LOG.md</code> in the repository.
      </CardContent></Card>
    </div>
  );
}
