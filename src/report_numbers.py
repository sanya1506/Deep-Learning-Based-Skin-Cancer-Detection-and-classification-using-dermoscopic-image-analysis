"""Collect every table produced by the notebooks into results/study/report_numbers.json,
so each number in the report can be traced to one file (report Section 1.5)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def collect():
    out = {}
    for f in sorted(config.STUDY_DIR.rglob("*")):
        if f.name == "report_numbers.json" or f.suffix not in (".csv", ".json"):
            continue
        key = f.relative_to(config.STUDY_DIR).as_posix()
        out[key] = json.load(open(f)) if f.suffix == ".json" else pd.read_csv(f).to_dict(orient="records")
    for f in sorted((config.RESULTS_DIR / "v2").rglob("*.json")):
        out["v2/" + f.relative_to(config.RESULTS_DIR / "v2").as_posix()] = json.load(open(f))
    runs = {}
    for f in sorted((config.RESULTS_DIR / "runs").glob("*/summary.json")):
        runs[f.parent.name] = json.load(open(f))
    out["training_runs"] = runs
    t = config.STUDY_DIR / "timings.json"
    out["notebook_minutes"] = json.load(open(t)) if t.exists() else {}
    json.dump(out, open(config.STUDY_DIR / "report_numbers.json", "w"), indent=1, default=str)
    print(f"{len(out)} entries -> {config.STUDY_DIR / 'report_numbers.json'}")
    return out


class timed:
    """with timed("14_imbalance_study"): ...  -> minutes stored in results/study/timings.json"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import time
        self.t0 = time.time()

    def __exit__(self, *a):
        import time
        f = config.STUDY_DIR / "timings.json"
        d = json.load(open(f)) if f.exists() else {}
        d[self.name] = round((time.time() - self.t0) / 60, 1)
        json.dump(d, open(f, "w"), indent=2)
