#!/usr/bin/env bash
# Copy real EDA / evaluation figures into the frontend so the Methodology page can show them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/frontend/public/figures"
mkdir -p "$DST"
RUN=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_name'])" "$ROOT/ml/artifacts/serving/model_card.json")
for f in class_distribution_plantvillage split_distribution rgb_distribution samples_plantdoc; do
  cp "$ROOT/ml/artifacts/eda/$f.png" "$DST/"
done
for f in model_comparison gradcam_examples; do
  cp "$ROOT/ml/artifacts/evaluation/$f.png" "$DST/"
done
for f in risk_coverage reliability_external_test_plantdoc confusion_matrix_test_normalized; do
  cp "$ROOT/ml/artifacts/evaluation/$RUN/$f.png" "$DST/"
done
echo "figures synced from run $RUN"
