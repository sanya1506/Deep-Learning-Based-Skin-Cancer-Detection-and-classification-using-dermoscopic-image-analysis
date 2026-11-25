"""Copy a handful of held-out TEST images (different classes) into sample_images/ for the demo.

If results/<model>/test_predictions.csv exists, one correctly classified image per class is chosen
(median confidence among correct predictions, so the demo is representative rather than cherry-picked
for maximum confidence). Otherwise a random test image per class is used.

Usage:
    python scripts/prepare_samples.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SAMPLE_CLASSES = ["mel", "bcc", "akiec", "nv", "bkl", "vasc"]


def main(model_name: str = config.DEFAULT_MODEL):
    test = pd.read_csv(config.SPLITS_DIR / "test.csv")
    preds_file = config.RESULTS_DIR / model_name / "test_predictions.csv"
    preds = pd.read_csv(preds_file) if preds_file.exists() else None

    for old in config.SAMPLE_DIR.glob("*.jpg"):
        old.unlink()
    for cls in SAMPLE_CLASSES:
        pick = None
        if preds is not None:
            ok = preds[(preds.dx == cls) & (preds.pred == cls)].sort_values("confidence")
            if len(ok):
                pick = ok.iloc[len(ok) // 2]["image_id"]
        if pick is None:
            pick = test[test.dx == cls].sample(1, random_state=config.SEED)["image_id"].iloc[0]
        src = config.RAW_DIR / Path(test.loc[test.image_id == pick, "path"].iloc[0])
        dst = config.SAMPLE_DIR / f"{cls}_{pick}.jpg"
        shutil.copy2(src, dst)
        print(f"{cls:6s} -> {dst.name}")


if __name__ == "__main__":
    main()
