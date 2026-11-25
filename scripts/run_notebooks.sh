#!/usr/bin/env bash
# Execute all training notebooks in order, saving outputs inside the .ipynb files.
set -e
cd "$(dirname "$0")/.."
JUPYTER="${JUPYTER:-./venv/bin/jupyter}"
for nb in notebooks/01_data_preprocessing_eda.ipynb notebooks/02_model_efficientnet_b0.ipynb \
          notebooks/03_model_resnet50.ipynb notebooks/04_model_densenet121.ipynb notebooks/05_model_comparison.ipynb; do
  echo "=== $(date '+%H:%M:%S') START $nb"
  "$JUPYTER" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
  echo "=== $(date '+%H:%M:%S') DONE  $nb"
done
echo "ALL NOTEBOOKS DONE"
