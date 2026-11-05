# Limitations and Risks

KrishiGuard AI is a portfolio/research prototype and a decision-support aid. It must not be the sole basis for
spraying, buying inputs or destroying crops.

## 1. Model scope
* **Closed set of 38 classes (14 crops).** Any other crop, disease, pest damage, nutrient deficiency, herbicide injury,
  abiotic stress or a *combination* of problems will be forced onto one of the 38 labels. The calibrated-confidence
  rejection ("not recognised" / "uncertain") reduces but does **not** eliminate confident wrong answers on such inputs;
  no out-of-distribution dataset was available to measure this.
* **One leaf, one label.** Whole plants, multiple leaves, fruit or stem symptoms are out of scope.
* **No severity estimation** and no localisation of lesions (Grad-CAM is an explanation, not a detector).

## 2. Domain shift — the most important limitation
* Training data (PlantVillage) are lab photos: single detached leaves on a uniform background, even lighting.
  On real field photos accuracy drops sharply (see the PlantDoc and Tomato-field rows in
  [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md)). Lab accuracy must not be quoted as field accuracy.
* PlantDoc labels are known to be noisy and some classes have very few test images (single digits), so per-class
  external numbers have wide uncertainty.
* The Tomato-PK field set was augmented by its authors (rotated copies), so its 7,200 images are correlated and its
  effective sample size is smaller.
* Training on labelled field photos (the prepared PlantDoc `domain_train` split, or Indian field images) is the main
  next step; a domain-adaptation run was planned but not executed.

## 3. Confidence and explanations
* Temperature scaling calibrates *on average* on the validation domains; calibration does not transfer reliably to
  new domains, cameras or regions. A high confidence can still be wrong.
* Thresholds were selected on validation data containing only PlantVillage and PlantDoc distributions.
* The blur detector is a single global sharpness statistic; it can miss motion blur on part of the image and can
  flag naturally smooth leaves.
* Grad-CAM shows which regions influenced the class score at a coarse resolution (7×7 for EfficientNet at 224 px);
  it is not evidence that the reasoning is agronomically correct.

## 4. Advisory content
* Compiled from cited public sources (ICAR/NIPHM, TNAU, Kerala Agricultural University, NHB, FAO and US university
  extension services); several entries rely only on non-Indian extension sources (listed in `advisory/SOURCES.md`).
* **Not yet reviewed by a practising agronomist**; Hindi text was written without native-speaker agronomic review.
* Pesticide registration in India changes (CIB&RC). Names are given only as "options reported by <source>" without
  dosages; every chemical list ends with the instruction to use only currently registered products, follow the label
  and consult the KVK / agriculture officer. Example: an older NIPHM package lists streptocycline for tomato bacterial
  spot, but the Government of India has moved to prohibit streptomycin + tetracycline in agriculture, so the app
  shows a "do not use" warning instead. Other actives (e.g. zineb) may also change status — always check.
* Regional practices (varieties, seasons, state recommendations) are not personalised.

## 5. Engineering limits
* Only one architecture (EfficientNet-B0) was trained; the planned ResNet-50 / ViT / EfficientNet-B2 comparison and
  the domain-adaptation run were dropped to keep the compute load suitable for a laptop.
* Trained on an 8 GB Apple M1: short epoch budgets, Optuna on a proxy task only for EfficientNet-B0, single seed
  (no confidence intervals across seeds), MPS kernels not bit-reproducible.
* CPU latency was measured on the development laptop while other workloads were running; production hardware differs.
* Rate limiting is in-memory per process (use Redis storage when scaling horizontally).
* Docker images were written and the compose file validated, but images were **not built or run** in this
  environment (Docker daemon not running, < 8 GB free disk).
* No authentication or user accounts; feedback is anonymous and unverified.

## 6. Data and ethics
* Uploaded images are not stored; only a SHA-256 hash and prediction metadata are kept. Hashes of identical images can
  still be linked across requests.
* User feedback is stored as `unverified` and must never be used as training labels without expert verification.
* Dataset licences: PlantVillage CC BY-SA 3.0 (share-alike obligations apply to redistributed derivatives of the
  data), PlantDoc and Tomato-PK CC BY 4.0 (attribution in the app footer and README).
