"""Condense text only (no change to headings, tables, figures, captions or styles). Every number is kept.
Run after update_report.py:  python condense.py Design_Project_13_Report_updated.docx"""
import re
import sys
import docx
from docx.oxml.ns import qn

path = sys.argv[1]
D = docx.Document(path)
P = [docx.text.paragraph.Paragraph(e, D) for e in D.element.body.iterchildren() if e.tag == qn("w:p")]
ZW = re.compile(r"[​-‍﻿]+")
log = []


def find(start):
    hits = [p for p in P if ZW.sub("", p.text).strip().startswith(start)]
    assert len(hits) == 1, f"{len(hits)} paragraphs start with {start!r}"
    return hits[0]


def condense(start, text):
    p = find(start)
    old = p.text
    tail = "".join(ZW.findall(old[-400:]))           # keep the invisible characters at the end
    runs = list(p._p.iter(qn("w:r")))
    text_runs = [r for r in runs if r.find(qn("w:t")) is not None and r.find(qn("w:drawing")) is None]
    keep = text_runs[0]
    for r in text_runs[1:]:
        r.getparent().remove(r)                      # runs holding a picture are not touched
    ts = keep.findall(qn("w:t"))
    for t in ts[1:]:
        keep.remove(t)
    lead = " " if ZW.sub("", old).startswith(" ") else ""
    ts[0].text = lead + text + tail
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    nums_old = set(re.findall(r"\d+(?:[.,]\d+)?", ZW.sub("", old)))
    nums_new = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    lost = sorted(nums_old - nums_new)
    log.append((start, len(ZW.sub("", old)), len(text), lost))


C = condense
# ------------------------------------------------------------------ literature review
C("Esteva et al. built", "Esteva et al. (2017) reached dermatologist-level performance with a convolutional neural network on a "
  "biopsy-proven test set. HAM10000 (Tschandl et al., 2018) has dermoscopic images of seven classes with lesion IDs and was the "
  "reference set of the ISIC 2018 challenge (Codella et al., 2019). The papers below are summarised in our own words; the figures "
  "are those reported by the authors.")
C("DSCC_Net:", "DSCC_Net is a CNN for four skin cancer classes (melanoma, basal cell carcinoma, squamous cell carcinoma and melanocytic "
  "nevus) trained on ISIC 2020, HAM10000 and DermIS with SMOTE-Tomek balancing and compared with six re-run backbones, including "
  "EfficientNet-B0. On the pooled test set it reports 94.17% accuracy, 99.43% AUC and 93.76% sensitivity.")
C("Detection of skin cancer using multi-scale", "A multi-scale MobileNetV2 classifies the full image and two central crops and combines "
  "the results to detect melanoma. We could not verify the reported metrics against the publication and therefore do not report them "
  "[VERIFY].")
C("A PeerJ Computer Science study", "MSCDNet combines DenseUNet segmentation, binary dragonfly feature selection and a SqueezeNet "
  "classifier on ISIC 2019. We could not check the reported numbers against the paper, so they are not included.")
C("Enhancing skin lesion detection:", "A two-stage CNN first separates benign from malignant lesions and then assigns the subclass, "
  "reusing the first-stage weights; on ISIC 2018 it reports 93.4% accuracy for the first stage and 96.2% for the second.")
C("Skin lesion analyser:", "A MobileNet pre-trained on ImageNet was fine-tuned for the seven HAM10000 classes and deployed as a public "
  "web app, reaching 83.1% categorical accuracy and a weighted F1-score of 0.83.")
C("Comparison of the performance of two deep", "VGG16 and ResNet50 were fine-tuned on HAM10000 with classes balanced by random "
  "resampling, reaching 93.45% and 94.69% accuracy; VGG16 trained much faster.")
C("Uncertainty-aware and explainable ensemble", "An ensemble of MaxViT-Tiny, ConvNeXt and EfficientNetV2-B0 with Monte Carlo dropout "
  "and Grad-CAM++ reports 96% accuracy, 99% ROC-AUC and 95% macro-F1 on HAM10000, but only on the predictions kept after uncertainty "
  "filtering (preprint).")
C("Multi-paradigm Vision Transformer ensemble", "Swin-Tiny, ViT-Base and DeiT-Small are combined by an attention module and an MLP "
  "meta-learner, trained with class-weighted loss and MixUp on HAM10000. The reported 98.37% accuracy, 98.39% weighted F1-score and "
  "0.999 mean AUC are the highest among the reviewed papers, at a high training and deployment cost.")
C("Transfer Learning Based Multi-Class", "A VGG16 with frozen early layers, class balancing and augmentation reached 94.2% accuracy and "
  "a macro-F1 of 0.88 on HAM10000; its baselines were taken from the literature, not re-run.")
C("MelanomaNet:", "MelanomaNet uses an EfficientNetV2 backbone with Grad-CAM++, automatic ABCDE criteria, concept activation vectors and "
  "Monte Carlo dropout on the 9 ISIC 2019 classes, reaching 85.61% accuracy and a weighted F1-score of 0.8564 (preprint).")
C("Adamson and Smith (Adamson", "Dermatology AI trained mainly on light skin may perform worse on darker skin and widen gaps in care "
  "(Adamson and Smith, 2018), and dark skin is rare in public dermoscopic datasets (Kinyanjui et al., 2020). The Fitzpatrick 17k "
  "clinical images, labelled with Fitzpatrick skin type (Fitzpatrick, 1988), showed that accuracy changes across skin types (Groh et al., "
  "2021), and published classifiers performed worse on the biopsy-confirmed DDI set, especially on dark skin (Daneshjou et al., 2022). "
  "The Monk scale (Monk, 2023) gives a finer ten-point description of skin tone.")

# ------------------------------------------------------------------ chapters 5 and 6
C("The Streamlit application app_v3.py", "The Streamlit application app_v3.py takes an uploaded or sample HAM10000 image and any single "
  "model, the ensemble (default) or the two-stage pipeline (meant for clinical photographs). It shows the image with the U-Net lesion "
  "outline, the crop, the Grad-CAM heatmap, the class probabilities and a PDF or CSV report; a threshold slider shows the validation "
  "sensitivity and specificity. A warning box gives the measured DermaCon-IN false-cancer rate and states that DDI was not evaluated, "
  "and a disclaimer says the output is not a diagnosis (screenshots in Appendix 2).")
C("Notebooks 09 and 19 run checks.", "Notebooks 09 and 19 check that no lesion or patient is in two subsets, that every selection is "
  "reproduced from validation data, that each experiment has three seeds and that every number is traced to its file. All 6 HAM10000 "
  "checks and all 16 Indian-study checks passed, with 0 failures.")
C("Table 6.1 shows all models", "Table 6.1 shows all models and ensembles on the 1,502 HAM10000 test images, with test-time augmentation. "
  "The weighted ensemble had the highest validation macro-F1 (0.755, against 0.742 for the best three models and 0.735 for equal "
  "weights) and was selected; on test it reached an accuracy of 0.846, a macro-F1 of 0.771 and a melanoma recall of 0.635. The "
  "best-three ensemble had a slightly higher test accuracy (0.848) but a lower macro-F1 (0.770). EfficientNet-B0 and its long run are "
  "identical because both peaked at epoch 10. All choices were made on validation data only.")
C("Melanoma is the weakest class", "Melanoma is the weakest class (recall 0.635, F1 0.637): 60 of 167 test melanomas were called benign "
  "(35.9%, 95% CI 29.0–43.4%), mostly nevus (49) or benign keratosis (11). Nevus is recognised best.")
C("On the 802 non-neoplastic", "On the 802 non-neoplastic DermaCon-IN test images every HAM10000 model raised false cancer alarms "
  "(Table 6.3): 11.0% for the ensemble, from 9.1% (EfficientNet-B0) to 29.2% (ConvNeXt-Tiny).")
C("Skin disorders such, as hair", "After the mixed group of other skin disorders, hair disorders, acne and pigmentary disorders "
  "(including vitiligo) drew the most alarms, and fungal and parasitic infections the fewest. These conditions may produce spots, sharp "
  "colour borders or raised lesions that resemble the tumour images, but this has not been tested. Groups with few images have wide "
  "intervals.")
C("On the 16 test images", "On the 16 test images with a HAM10000 class, accuracy ranged from 0.0% to 18.8%: on photographs the models "
  "do not recognise even the lesion types they were trained on. Across all images the ensemble found 26.3% of the malignant images and "
  "none of the 5 test basal cell carcinomas. ConvNeXt-Tiny found the most cancers but also raised the most false alarms; no model did "
  "both well.")
C("Stage 1 gate, the classifier", "The Stage 1 gate reaches a test accuracy of 0.592 ± 0.013 and a macro-F1 of 0.422 ± 0.016 on the "
  "eight main classes. Because the threshold must keep the validation neoplasms, it ends up very low (Table 6.7).")
C("The gate lowered the FCR", "The gate lowered the FCR from 11.0% to 5.6% on average, and the interval of the change does not include "
  "zero. The per-seed rates were 9.7%, 3.1% and 3.9%, so the benefit depends strongly on the seed (Figure 6.6 shows the effect per "
  "class). Whether the gate also blocks real cancers cannot be measured here; the ensemble missed all test basal cell carcinomas even "
  "without the gate.")
C("The gate was trained using photographs.", "The gate was trained on photographs: on the HAM10000 dermoscopic validation images it "
  "identifies only 30.9% of lesions and 27.9% of malignant ones as neoplasms (seed 2, the best by validation macro-F1). The two-stage "
  "option is therefore meant for clinical photographs, as the application states; Appendix 2 shows it rejecting a melanoma.")
C("DDI could not be obtained (registration", "DDI could not be obtained (registration on Redivis failed), so this evaluation has not "
  "been run and its numbers are [PENDING: DDI]; notebook 13 runs automatically once the images and ddi_metadata.csv are placed in "
  "data/ddi/. Without DDI the project has no test of cancer detection on darker skin with enough biopsy-confirmed cancers.")
C("The rule chose focal loss.", "The rule chose focal loss: its validation macro-F1 (0.688) was within 0.005 of no weighting (0.687), and "
  "the tie-break went to its higher validation melanoma recall (0.572 vs 0.566). On test the weighted sampler had the highest macro-F1 "
  "(0.689 vs 0.675), but the intervals overlap, and the choice was not changed. The sampler also gave the highest melanoma recall "
  "(0.703) and inverse-frequency weights the highest malignant sensitivity (0.751), both at about 0.755 accuracy. The single "
  "EfficientNet-B0 of Table 6.1 (seed 42, with test-time augmentation) reached a macro-F1 of 0.708, above the three-seed mean of the "
  "same recipe (0.679), which shows how much one run can vary.")
C("On HAM10000 the augmentation did not help", "On HAM10000 the augmentation did not help: melanoma recall fell by 0.094 (interval "
  "−0.148 to −0.038), malignant specificity rose slightly (+0.017) and every other interval includes zero. On DermaCon-IN it raised "
  "the FCR from 16.3% ± 7.2% to 19.8% ± 13.9% and called on average 4.7 instead of 2.7 of the 5 test basal cell carcinomas cancer: the "
  "augmented model says cancer more readily on photographs in both directions and is not clearly better.")
C("The U-Net does a good job", "The U-Net segments HAM10000 lesions well (Dice 0.949 vs 0.691 for Otsu thresholding). Training on crops "
  "improved macro-F1 (+0.0485), melanoma recall (+0.1597) and malignant sensitivity (+0.1080), all with intervals excluding zero, while "
  "malignant specificity fell (−0.0289) and accuracy did not change clearly: cropping trades some false alarms for fewer missed cancers, "
  "which suits a screening tool. On DermaCon-IN the crop model's call rate was 15.7% ± 3.0%; the dermoscopy-trained U-Net missed the "
  "lesion in some photographs, which then fell back to the full image.")
C("All models lose accuracy when corruption", "All models lose accuracy under strong corruption; Swin-T and ConvNeXt-Tiny degrade least. "
  "JPEG compression hurts cancer detection most: ensemble malignant sensitivity drops from 69.3% to 18.6% at the lowest quality and to "
  "34.1% at the strongest darkening, which matters for often compressed and poorly lit phone photographs. The clean numbers here come "
  "from re-preprocessed test images and differ slightly from Table 6.1, which used cached images.")
C("For each model Grad-CAM provides", "For every model Grad-CAM gives a lower deletion score and a higher insertion score than a random "
  "map, so the heatmaps highlight regions the model actually uses, though not necessarily the clinically relevant ones.")
C("To see how far the HAM10000 result", "A second pipeline (v2) was trained separately to see how far the HAM10000 result can be pushed. "
  "It uses EfficientNet-B4 (Noisy Student), EfficientNetV2-S (ImageNet-21k) and ConvNeXt-Small (ImageNet-22k) at 384 × 384 pixels, "
  "U-Net lesion crops (trained on the ISIC 2018 masks), the patient's age, sex and body site, and about 16,000 extra ISIC 2019 training "
  "images from which every HAM10000 image and every image of a validation or test lesion was removed. Each model was trained for 18 "
  "epochs and predicts with 8-view test-time augmentation. Ensemble weights, a per-class bias and the malignant threshold were fitted on "
  "validation, which gave ConvNeXt-Small a weight of zero: the final ensemble combines EfficientNet-B4 (0.55) and EfficientNetV2-S "
  "(0.45). Training took 9.1 hours on Kaggle.")
C("On the same HAM10000 test split (1,502", "On the same 1,502 test images v2 reaches 90.35% accuracy (95% CI 88.3–92.3%), a macro-F1 of "
  "0.831 and 93.1% benign-vs-malignant accuracy (Table 6.17); the members alone reach 89.7% (EfficientNet-B4), 88.9% (EfficientNetV2-S) "
  "and 87.5% (ConvNeXt-Small). With extra training data and 384-pixel inputs, v2 is not directly comparable to the six 224-pixel models "
  "of Table 6.1.")
C("Most of the gain comes from the common", "The gain comes mostly from common classes (nevus recall 0.979, basal cell carcinoma 0.986); "
  "melanoma recall (0.641) stays close to that of the selected ensemble (0.635). Malignant sensitivity is 76.6% at 97.1% specificity, or "
  "95.2% at 83.3% at the threshold chosen on validation for 90% sensitivity. v2 was not evaluated on DermaCon-IN; Figure 6.10 shows its "
  "Grad-CAM maps.")
C("The HAM10000 ensemble is a reasonable", "The HAM10000 ensemble is a reasonable dermoscopic classifier, yet it misses about 35.9% of "
  "melanomas. On Indian clinical photographs it fails in two ways: it calls 11.0% of common non-tumour conditions cancer, and it does "
  "not recognise the few tumours it should know. The alarms do not depend on Fitzpatrick type within DermaCon-IN, so the main causes "
  "appear to be the change from dermoscopy to ordinary photographs and conditions the model has no class for; a gate trained on Indian "
  "data reduces the alarms, which supports the second explanation, but the gate itself is weak.")
C("This project trained six learning models", "This project trained six models and a validation-weighted ensemble for seven-class skin "
  "lesion classification on HAM10000 (test macro-F1 0.771) and built a web application around them; a separately trained improved "
  "pipeline (v2: 384-pixel inputs, extra ISIC 2019 data, U-Net crops and patient metadata) reached 90.35% accuracy on the same test "
  "split, though not a higher melanoma recall. Tested without retraining on DermaCon-IN, the models called 11.0% of non-tumour Indian "
  "skin photographs cancer, recognised few of the lesion types they were trained on and missed most of the few Indian cancers; a gate "
  "trained on Indian data lowered the false-cancer rate to 5.6%. The supporting studies, each repeated with three seeds, showed no "
  "benefit from skin-tone augmentation, a gain in cancer sensitivity at some cost in specificity from lesion cropping and a sharp loss "
  "of cancer detection under JPEG compression. Good results on HAM10000 do not mean good results for Indian patients.")

D.save(path)
tot_old = sum(x[1] for x in log); tot_new = sum(x[2] for x in log)
print(f"{len(log)} paragraphs condensed: {tot_old} -> {tot_new} characters ({100 * (1 - tot_new / tot_old):.0f}% shorter)")
for s, a, b, lost in log:
    if lost:
        print("numbers not repeated:", s, lost)
with open(path.replace(".docx", "_condense_log.txt"), "w", encoding="utf-8") as f:
    for s, a, b, lost in log:
        f.write(f"{s!r}: {a} -> {b} chars; numbers dropped: {lost}\n")
