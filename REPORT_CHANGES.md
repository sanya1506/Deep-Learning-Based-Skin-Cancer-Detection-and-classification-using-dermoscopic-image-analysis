# Changes in Design_Project_13_Report_updated.docx

The original (`Downloads\Design Project 13 Report.docx`) is unchanged. In the copy, the front matter (title pages,
certificate, declaration, acknowledgement) is unchanged, and fonts, styles and layout are kept. Only text inside existing
runs and table cells, and the figures, were changed. 62 pages (original 61).

## New results (Parts A and B, Kaggle, Nov 2026)
- Table 3.1: same split, numbers identical.
- Table 6.1 (all 9 rows), Section 6.1 text (rewritten: the selected weighted ensemble is now also best on test macro-F1;
  EfficientNet-B0 and its long run are identical because both peak at epoch 10). "AUC" column = macro AUC.
- Table 6.2, the melanoma paragraph under it, Table 6.16, Figures 6.1, 6.2, 6.3, 6.4, 6.9.
- Table 5.4 (epochs and minutes per model); Table 5.5 rows for the imbalance, skin-tone and segmentation notebooks.
- Table 6.10 and its text (rewritten: the rule now selects focal loss, not the weighted sampler), Figure 6.7.
- Table 6.11 and the HAM10000 sentence of 6.6 (rewritten: melanoma recall now falls clearly with the augmentation).
- Tables 6.12, 6.13 and the HAM10000 part of 6.7 (rewritten: cropping now raises cancer sensitivity), Figure 4.4.
- Discussion (35.9% missed melanomas; cropping clause), Limitations (imbalance sentence), Conclusion (macro-F1 0.771, cropping clause).
- Abstract: macro-F1 0.771 and accuracy 84.6% filled in.

## New section 6.10 Improved pipeline (v2)
- Method, results (90.35% accuracy, 95% CI 88.3–92.3%, macro-F1 0.831, benign-vs-malignant 93.1%), new Table 6.17 (per class)
  and Figure 6.10 (Grad-CAM). Discussion → 6.11, Limitations → 6.12. One sentence on v2 in the Conclusion.
- Contents, list of tables and list of figures: new rows; every page number recomputed from the Word layout.
- Note: you quoted 93.0% for v2 benign-vs-malignant accuracy; the saved metrics give 93.1% for the calibrated ensemble
  (93.0% is the uncalibrated one), so 93.1% is used, consistent with the 90.35% figure.

## Implementation chapter
- Table 5.1 hardware → Kaggle cloud notebook, 4 vCPU, 29 GB RAM, NVIDIA Tesla T4 16 GB (CUDA), Linux. Sentence under it updated.
  Please check the CPU/RAM line against your Kaggle session if you want it exact.
- Table 5.2: Python 3.13 and PyTorch 2.11.0 (as printed in the Kaggle logs). The other package versions were not changed.

## Missing numbers / words filled in (no new results)
- Abstract: 5,450 photos, 802 non-cancer test images, 11.0%, 5.6% (from the report's own Tables 3.2, 6.3, 6.8).
- Chapter 1 citation years (Esteva 2017, Tschandl 2018, Codella 2019, Adamson and Smith 2018, Kinyanjui 2020, Groh 2021,
  Daneshjou 2022, Madarkar 2025); "HAM00"/"HAM10" → HAM10000 (Ch. 1 and 2.1).
- "802 -neoplastic" → "802 non-neoplastic"; "(, with seed 2" → "(seed 2,"; "DermaCon-, IN" → "DermaCon-IN"; "reported ass" →
  "reported as is"; "-tumour" → "non-tumour"; "trained on data" → "trained on Indian data"; "do not mean results" → "do not mean good results".

## Kept from the old report (Part C is not run yet)
- Tables 3.2–3.6, 6.3–6.9, 6.14, 6.15, Figures 3.3, 6.5, 6.6, 6.8, the DermaCon-IN sentences in 6.2–6.9, Table 5.5 rows
  for notebooks 10, 11, 12, 17 and 18, Section 5.6 sanity text, and Appendix 2 screenshots.
- When the Kaggle Part C run finishes, these can be replaced the same way.

## Not changed (front matter rule)
- Acknowledgement still says "HAM00" (should be HAM10000).

The full cell-by-cell list is in `kaggle_runs/report/changes.md`.

## Condensed (requested 23:40)
- 33 paragraphs condensed (literature review 2.1.1–2.1.10, 2.3, Sections 5.5, 5.6, 6.1–6.11, Conclusion): 30% fewer characters.
  No heading, table, figure, caption or style was changed and no result number was removed (only publication details such as
  volume numbers, which are still in Table 2.1 and the references). Per-paragraph log: `kaggle_runs/report/Design_Project_13_Report_updated_condense_log.txt`.
- Result: 61 pages (was 62). Tables, figures and chapter page breaks set most of the length.
