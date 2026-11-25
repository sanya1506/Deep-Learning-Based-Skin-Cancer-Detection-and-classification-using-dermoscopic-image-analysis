# v2 – Improved pipeline (target: ≥ 90 % accuracy)

The v1 models (224 px, torchvision backbones, 12–15 epochs, HAM10000 only) reach **82.0 %**
7-class accuracy and **87.6 %** benign/malignant accuracy on the leakage-free test split.
v2 keeps that **exact same test split** and changes the parts of the pipeline that limit accuracy.

## Techniques added

| # | Technique | Where | Why it helps |
|---|---|---|---|
| 1 | **Extra data: ISIC 2019** (~16 k extra labelled dermoscopic images, AK/SCC → akiec) | `common.py` | Data quantity is the largest single factor on this task. Strict leakage filters drop every HAM10000 image and every image of a HAM val/test lesion. |
| 2 | **Lesion segmentation (U-Net, EfficientNet-B4 encoder)** trained on ISIC 2018 Task 1 masks, then **crop to lesion + 25 % margin** | `segmentation.py` | The network sees the lesion at higher effective resolution instead of skin/vignette/rulers. Otsu fallback if you skip the U-Net. |
| 3 | **Shades-of-Gray colour constancy** (replaces CLAHE) | `common.py` | Normalises the illumination of different dermatoscopes without distorting lesion colour, which is diagnostic. CLAHE + hair inpainting are no longer used. |
| 4 | **384 px input** (was 224) | `train_cls.py` | Fine texture, dots/globules and border detail survive. |
| 5 | **Stronger pretrained backbones**: EfficientNetV2-S (IN-21k), ConvNeXt-S (IN-22k, 384), EfficientNet-B4 Noisy-Student | `train_cls.py` | Better transfer features than ImageNet-1k ResNet50/DenseNet121/EffNet-B0. |
| 6 | **Metadata fusion**: age, sex, anatomical site → MLP concatenated with image features (with metadata dropout) | `train_cls.py` | Site/age are strong priors (e.g. bkl/akiec on older face skin, nv on young back). Typically +1–2 %. |
| 7 | **Training recipe**: mixup/cutmix, label smoothing, RandomResizedCrop + 180° rotation + shear + colour jitter + random erasing, AdamW, warm-up + cosine, AMP, **EMA weights**, 25 epochs | `train_cls.py` | Less overfitting, better-trained models (v1 stopped at 12–15 epochs). |
| 8 | **8-view test-time augmentation** (flips × 90° rotations) | `train_cls.py`, `predict_v2.py` | Dermoscopy has no canonical orientation. |
| 9 | **Ensemble of 3 architectures** with weights fitted on validation NLL | `ensemble.py` | Different architectures make different errors. |
| 10 | **Prior/bias calibration on validation** (per-class logit bias maximising val accuracy) and a validation-chosen benign/malignant threshold | `ensemble.py` | Corrects the class-prior shift from extra data and imbalance. Nothing is tuned on test. |
| 11 | *(reported separately)* **lesion-level averaging** | `ensemble.py` | HAM10000 often has 2–6 images of the same lesion; averaging them is valid only when lesion identity is known. |

## Measured on the existing v1 predictions (no retraining)

| Setting | 7-class acc | Binary acc* |
|---|---|---|
| EfficientNet-B0 (v1 best) | 82.0 % | 87.8 % |
| Mean of the 3 v1 models | 83.2 % | 89.4 % |
| + prior correction + lesion averaging | 84.0 % | 89.8 % |

*Binary = summed malignant probability ≥ 0.5, which is why the B0 figure differs slightly from the 87.6 % in `model_comparison.csv`.

So ensembling alone cannot reach 90 % with the v1 models; techniques 1–8 (retraining) are needed.

## v2 results (Kaggle T4 run, 9.1 h, same lesion-level HAM10000 test split as v1)

| HAM10000 test (1,502 images) | v1 EfficientNet-B0 | v2 ensemble |
|---|---|---|
| 7-class accuracy | 82.0 % | **90.35 %** (95 % lesion-bootstrap CI 88.3-92.3 %) |
| Macro-F1 | 0.718 | **0.831** |
| Balanced accuracy | 0.719 | 0.810 |
| Benign vs malignant accuracy | 87.6 % | **93.0 %** (CI 91.2-94.6 %) |
| Malignant sensitivity / specificity (default threshold) | 0.652 / - | 0.772 / 0.968 |
| Malignant sensitivity / specificity (threshold 0.14, chosen on val for >= 90 % sens.) | - | 0.952 / 0.833 |
| Melanoma recall | 0.527 | 0.641 |
| Macro ROC-AUC | - | 0.982 |

Individual models (test accuracy): EfficientNet-B4-NS 89.7 %, EfficientNetV2-S 88.9 %, ConvNeXt-S 87.5 %.
The validation-fitted ensemble kept EfficientNet-B4 (weight 0.55) and EfficientNetV2-S (0.45) with small class
biases (mel -0.20, bcc -0.15, vasc +0.80). With lesion-level averaging of repeated images: 90.8 %.
Training used ISIC 2019 extra images (leakage-filtered), U-Net crops, metadata, 384 px, 18 epochs.

## How to run (GPU required: Kaggle / Colab)

Easiest path: open `notebooks/06_improved_pipeline_v2.ipynb` on Kaggle or Colab. Equivalent commands:

```bash
pip install timm segmentation-models-pytorch
# 1. segmentation (optional, else use --no-unet below)
python improved/segmentation.py train --task1-images <ISIC2018_Task1-2_Training_Input> --task1-masks <ISIC2018_Task1_Training_GroundTruth>
# 2. crop + colour constancy cache (HAM10000 must be in data/raw as before)
python improved/segmentation.py cache --isic2019 <ISIC2019_DIR>        # add --no-unet for the Otsu crop
# 3. classifiers
python improved/train_cls.py --arch tf_efficientnetv2_s.in21k_ft_in1k   --meta --isic2019 <ISIC2019_DIR>
python improved/train_cls.py --arch convnext_small.fb_in22k_ft_in1k_384 --meta --isic2019 <ISIC2019_DIR>
python improved/train_cls.py --arch tf_efficientnet_b4.ns_jft_in1k      --meta --isic2019 <ISIC2019_DIR>
# 4. ensemble + calibration (val only) and final test metrics
python improved/ensemble.py --search-subsets
```

`<ISIC2019_DIR>` must contain `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`
and the images (any sub-folder). Downloads: https://challenge.isic-archive.com/data/

Outputs: `models/v2/*.pth`, `results/v2/<run>/` (history, val/test predictions),
`results/v2/ensemble/` (metrics JSON, confusion matrix, classification report, config used by the app).

## Using v2 in the Streamlit app

`improved/predict_v2.py` returns the same dictionary as `src/predict.py`, including the Grad-CAM overlay. In `app.py`:

```python
from improved.predict_v2 import EnsemblePredictor
def get_predictor(model_name):           # keep the @st.cache_resource decorator
    return EnsemblePredictor()
```

`predict_array(img, age=None, sex=None, localization=None)` accepts the optional metadata. Without it, the
model falls back to "unknown", which it was trained to handle.

## Other ideas if you still need more

* 5-fold cross-validation on HAM train+val with fold ensembles (≈ +1 pt, 5× training cost)
* Larger inputs (448–512 px) or larger models (EfficientNet-B5/B6, ConvNeXt-Base, EVA-02, Swin-B at 384)
* Pseudo-labelling of the ISIC 2018/2019 *test* images (unlabelled) with the ensemble
* Knowledge distillation of the ensemble into one EfficientNetV2-S for a faster app
