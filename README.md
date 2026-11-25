# Deep Learning-Based Skin Cancer Classification and Its Evaluation on Indian Skin Images

Design Project – I, Alliance University. Research and teaching use only – not a diagnostic tool.

## What is in here
| Part | Code | Notebooks | Report |
|---|---|---|---|
| HAM10000: six models + validation-chosen ensemble | `src/ham.py`, `src/engine.py`, `src/models.py` | 01–09 | Ch. 3.1, 4.2–4.4, 6.1 |
| Indian skin (DermaCon-IN): audit, zero-shot false-cancer rate, Stage 1 gate, DDI | `src/dermacon.py` | 10–13 | Ch. 3.2, 4.5–4.6, 6.2–6.4 |
| Supporting studies: imbalance, skin tone, U-Net + crops, robustness, Grad-CAM quality | `src/studies.py` | 14–18 | Ch. 4.7, 6.5–6.9 |
| Statistics (lesion/patient bootstrap, Wilson, paired CIs), sanity checks, numbers file | `src/stats.py`, `src/sanity.py`, `src/report_numbers.py` | 09, 19 | Ch. 4.8, 5.6 |
| v2 high-accuracy pipeline (ISIC 2019 extra data, U-Net crops, 384 px, metadata, ensemble) – **90.4 % accuracy** | `improved/` | 20 | new section |
| Web apps | `app.py` (v1), `app_v3.py` (all models, ensemble, two-stage, v2) | | Ch. 5.5, App. 2 |

Every number ends up in `results/study/report_numbers.json` (v2: `results/v2/`).

## Running it (Kaggle, free GPU)
The `notebooks/kaggle/` notebooks contain all the code and the exact lesion-level split; nothing else to upload.
Settings for each: Accelerator **GPU T4 x2**, Internet **On**. Run each first with `SMOKE = True` (~15–20 min),
then with `SMOKE = False`.

1. **Part A** – six HAM10000 models + ensemble (~3 h). Input: `skin-cancer-mnist-ham10000`.
2. **Part B** – imbalance / skin-tone / U-Net + crop studies (~5–6 h). Inputs: HAM10000 + a HAM10000 lesion-mask
   dataset (otherwise downloaded from Harvard Dataverse). Can run at the same time as A.
3. **Part C** – robustness, Grad-CAM quality, DermaCon-IN, gate, DDI, sanity checks (~2 h). Inputs: HAM10000, your
   DermaCon-IN copy as a **private** Kaggle dataset, and the outputs of Parts A and B (Add Input → Your Work → Notebooks).

Download `part_A/B/C_outputs.zip` and unzip them into this folder (they contain `models/`, `results/`, `configs/`).
Locally, the same steps are notebooks 01–19 (`pip install -r requirements.txt`).

## App
```
pip install -r requirements.txt
streamlit run app_v3.py
```
For the v2 option, put the four `.pth` files from `v2_outputs.zip` into `models/v2/`.
