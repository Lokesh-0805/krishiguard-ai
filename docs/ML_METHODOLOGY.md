# ML Methodology

## 1. Problem formulation
Single-label image classification over the 38 PlantVillage classes (`crop___condition`, 26 diseases + 12 healthy,
14 crops). Crop identity is derived by summing class probabilities per crop (`crop_confidence`).
"Unsupported plant / not a leaf" has no training class; it is handled by calibrated-confidence rejection (§6), which
is a heuristic, not a guarantee (see LIMITATIONS).

## 2. Data protocol (details in DATASET_REPORT.md)
| Split | Source | Used for |
|---|---|---|
| train | PlantVillage groups (≈70 %) | gradient updates |
| validation | PlantVillage groups (≈15 %) | early stopping, Optuna objective, temperature, thresholds, model selection |
| external_validation | 20 % of PlantDoc-train groups | temperature, thresholds, model selection (never used for gradient updates) |
| domain_train | 80 % of PlantDoc-train groups | gradients **only** in the labelled domain-adaptation run |
| test | PlantVillage groups (≈15 %) | final internal metrics, evaluated once |
| external_test | PlantDoc-test (de-duplicated) + Tomato-PK field set | final external metrics, evaluated once |

Grouping = union of the official PlantVillage leaf-identity map, MD5 duplicates and pHash (≤ 4 bits) near-duplicates.

## 3. Models
All from `timm`, ImageNet-pretrained, new linear head for 38 classes, dropout 0.2 before the head.

| Model | timm id | Params | Input | Why |
|---|---|---:|---:|---|
| EfficientNet-B0 | `efficientnet_b0` | 4.1 M | 224 | efficient CNN baseline, cheap CPU serving |
| ResNet-50 | `resnet50` | 23.6 M | 224 | classic transfer-learning reference |
| ViT-S/16 | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 21.7 M | 224 | transformer comparison (ImageNet-21k pre-training) |
| EfficientNet-B2 | `efficientnet_b2` | 7.8 M | 256 | larger EfficientNet |
| EfficientNet-B0 + domain adaptation | init from the B0 run | 4.1 M | 224 | tests whether a little field data closes the domain gap |

> **What was actually trained:** only **EfficientNet-B0** (full 5-epoch run, deployed). The other rows were planned
> and are supported by the scripts, but were not trained after the project scope was reduced to a laptop-friendly
> college submission (see EXPERIMENT_LOG). A 1-epoch ResNet-50 learning-rate probe is the only ResNet result.

## 4. Training recipe (`ml/training/train.py`)
* AdamW; **differential learning rates**: head `lr`, backbone `lr × backbone_lr_mult`.
* Cosine schedule with linear warm-up (5 % of steps; 2 % for the adaptation run), stepped per batch.
* Loss: class-weighted (inverse √frequency, mean-normalised), label-smoothed cross-entropy implemented with
  gather/sum (`smoothed_weighted_ce`) — mathematically identical to PyTorch's but avoids an MPS sync/hang.
* Batch 32, fp32 on MPS (AMP only on CUDA), 3 DataLoader workers, seeded (Python/NumPy/torch, seed 42).
* Every epoch: PlantVillage-validation and PlantDoc-external-validation accuracy / macro-F1 / weighted-F1 are logged
  to MLflow (`ml/experiments/mlflow.db`, view with `mlflow ui --backend-store-uri sqlite:///ml/experiments/mlflow.db`).
* **Early stopping** on validation macro-F1 (patience 2, min delta 1e-4); best checkpoint kept. The adaptation run
  selects on the mean of validation and external-validation macro-F1 (it is the run meant to improve field data).
* Epoch budgets are small (3–5) because of the hardware (RESOURCE_REPORT.md). PlantVillage fine-tuning converges
  within a few epochs; curves are inspected for under/over-fitting in MODEL_EVALUATION_REPORT.md.

## 5. Hyper-parameter search
* **EfficientNet-B0 — Optuna** (TPE sampler, seed 42, 4 random start-up trials, 8 trials, sqlite study
  `ml/experiments/optuna.db`). Search space: head LR 3e-4–5e-3 (log), backbone multiplier 0.03–0.5 (log), weight decay
  1e-5–5e-2 (log), label smoothing 0–0.15. Proxy task: 1 epoch on a stratified 4,000-image train subset, scored by
  macro-F1 on a 1,500-image stratified validation subset. Limitation: it rewards fast convergence; it cannot see
  long-schedule effects. Best params are reused for EfficientNet-B2.
* **ResNet-50 — learning-rate probe**: head LR ∈ {1e-3, 3e-3} (backbone × 0.1), 1 epoch on 6,000 images; the better
  one is used for the full run.
* **ViT-S/16**: standard ViT fine-tuning recipe (head 1e-3, backbone 5e-5, weight decay 0.05) — not tuned (budget).

## 6. Calibration and uncertainty (`ml/evaluation/evaluate.py`)
1. **Temperature scaling**: a single T minimising the *mean of the per-domain NLLs* on PlantVillage-validation and
   PlantDoc-external-validation (grid + golden-section search). Equal domain weighting stops the 7k lab images from
   dominating the 463 field images.
2. **Confident threshold τ**: the lowest threshold at which calibrated selective accuracy is ≥ 90 % on **both**
   validation domains; if unattainable, the threshold that maximises the domain-averaged selective accuracy at ≥ 10 %
   coverage (the rule actually used is written into `metrics.json` and the model card).
3. **Reject threshold τ_r**: min(τ, 2nd percentile of calibrated confidence over correctly-classified validation
   images). Below τ_r the API answers *unrecognized*.
4. **Blur**: variance-of-Laplacian on a 256×256 grey copy; threshold = 1st percentile over validation images. Blurry
   photos are never reported as *confident*.
Metrics reported: ECE (15 bins) before/after scaling, reliability diagrams, risk–coverage curves and AURC.

## 7. Model selection (`ml/evaluation/compare_and_export.py`)
Score = mean(validation macro-F1, external-validation macro-F1); runs within 0.005 of the best are tie-broken by CPU
latency. No test or external-test number influences selection. The selected checkpoint is exported as a plain
`state_dict` (`ml/artifacts/serving/model.pt`, loaded with `weights_only=True`) plus `model_card.json`.

## 8. Explainability
Grad-CAM (Selvaraju et al., 2017) on the predicted class: EfficientNet `bn2` (post-`conv_head` activation), ResNet
`layer4`, ViT `blocks[-1].norm1` tokens reshaped to the 14×14 patch grid. Computed on the exact centre crop the model
sees and shown as model-input / heatmap / overlay. Presented as a sanity check, not as proof.

**Serving:** the API runs the model with ONNX Runtime (`ml/evaluation/export_onnx.py`). The ONNX graph also outputs
class activation maps (CAM = classifier weights × last feature map); because EfficientNet-B0 ends in global average
pooling + one linear layer, CAM equals Grad-CAM up to a constant factor. The export script verifies ONNX vs PyTorch
(identical predictions on 105 test images, max logit difference 1e-5) and CAM vs Grad-CAM (correlation > 0.9999).

## 9. Evaluation metrics
Accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1/support, raw and row-normalised
confusion matrices, top-3 accuracy, crop-level accuracy, ECE, selective accuracy & coverage at τ, unrecognized rate,
batch-1 latency (CPU 8 threads; MPS) and checkpoint size. Macro averages on external sets use only classes present in
that set (predictions into absent classes still count as errors).

## 10. Reproducibility
```bash
python -I ml/preprocessing/build_dataset.py
python ml/training/run_experiments.py               # resumable queue: tune → B0 → R50 → ViT → B0-domain → B2
python ml/evaluation/evaluate.py --run <run_name>    # for each run
python ml/evaluation/compare_and_export.py
```
Seeds are fixed; MPS kernels are not bit-deterministic, so re-runs can differ in the last decimal places.
