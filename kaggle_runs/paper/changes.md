Table 3 r1 c1: '0.820' → '0.825'
Table 3 r1 c2: '0.718' → '0.708'
Table 3 r1 c3: '0.527' → '0.599'
Table 3 r1 c4: '0.652' → '0.669'
Table 3 r1 c5: '0.931' → '0.935'
Table 3 r2 c1: '0.830' → '0.825'
Table 3 r2 c2: '0.717' → '0.708'
Table 3 r2 c3: '0.581' → '0.599'
Table 3 r2 c4: '0.693' → '0.669'
Table 3 r2 c5: '0.938' → '0.935'
Table 3 r3 c1: '0.808' → '0.833'
Table 3 r3 c2: '0.695' → '0.719'
Table 3 r3 c3: '0.593' → '0.587'
Table 3 r3 c4: '0.710' → '0.669'
Table 3 r3 c5: '0.951' → '0.946'
Table 3 r4 c1: '0.796' → '0.803'
Table 3 r4 c2: '0.643' → '0.680'
Table 3 r4 c3: '0.425' → '0.653'
Table 3 r4 c4: '0.545' → '0.721'
Table 3 r4 c5: '0.946' → '0.945'
Table 3 r5 c1: '0.826' → '0.827'
Table 3 r5 c2: '0.735' → '0.736'
Table 3 r5 c3: '0.557' → '0.581'
Table 3 r5 c4: '0.666' → '0.717'
Table 3 r5 c5: '0.936' → '0.945'
Table 3 r6 c1: '0.824' → '0.826'
Table 3 r6 c2: '0.677' → '0.699'
Table 3 r6 c3: '0.671' → '0.587'
Table 3 r6 c4: '0.738' → '0.693'
Table 3 r6 c5: '0.926' → '0.942'
Table 3 r7 c1: '0.848' → '0.846'
Table 3 r7 c2: '0.750' → '0.760'
Table 3 r7 c3: '0.635' → '0.611'
Table 3 r7 c4: '0.717' → '0.724'
Table 3 r7 c5: '0.966' → '0.965'
Table 3 r8 c1: '0.838' → '0.846'
Table 3 r8 c2: '0.747' → '0.771'
Table 3 r8 c3: '0.563' → '0.635'
Table 3 r8 c4: '0.672' → '0.741'
Table 3 r8 c5: '0.964' → '0.966'
P54: 'The weighted ensemble was chosen because it had the validation macro-F1 score of 0.760. In the test set the equal-weight ensemble achieved a higher macro-F1 of 0.750 but the weighted ensemble remained the choice we made. Unfortunately, the weighted ensemble still misclassified 43.1 % of the test melanomas with a 95 % confidence interval of 35.8 %, to 50.7 % labeling them as benign.' → 'The weighted ensemble was chosen because it had the highest validation macro-F1 (0.755). On the test set it also had the higher macro-F1 (0.771 vs 0.760) and melanoma recall (0.635 vs 0.611) of the two ensembles. Even so, it labelled 35.9 % of the test melanomas benign (95 % confidence interval 29.0 % to 43.4 %).'
Table 5 r1 c0: 'None' → 'None' (bold=False)
Table 5 r1 c1: '0.678' → '0.687' (bold=False)
Table 5 r1 c2: '0.683 ± 0.023' → '0.671 ± 0.018' (bold=False)
Table 5 r1 c3: '0.597 ± 0.075' → '0.501 ± 0.052' (bold=False)
Table 5 r2 c0: '√ weights (default)' → '√ weights (default)' (bold=False)
Table 5 r2 c1: '0.679' → '0.679' (bold=False)
Table 5 r2 c2: '0.695 ± 0.023' → '0.679 ± 0.005' (bold=False)
Table 5 r2 c3: '0.643 ± 0.052' → '0.657 ± 0.059' (bold=False)
Table 5 r3 c0: 'Inverse frequency' → 'Inverse frequency' (bold=False)
Table 5 r3 c1: '0.661' → '0.643' (bold=False)
Table 5 r3 c2: '0.650 ± 0.015' → '0.652 ± 0.027' (bold=False)
Table 5 r3 c3: '0.739 ± 0.035' → '0.669 ± 0.038' (bold=False)
Table 5 r4 c0: 'Focal loss' → 'Focal loss' (bold=True)
Table 5 r4 c1: '0.663' → '0.688' (bold=True)
Table 5 r4 c2: '0.649 ± 0.033' → '0.675 ± 0.020' (bold=True)
Table 5 r4 c3: '0.601 ± 0.056' → '0.539 ± 0.036' (bold=True)
Table 5 r5 c0: 'Weighted sampler' → 'Weighted sampler' (bold=False)
Table 5 r5 c1: '0.684' → '0.673' (bold=False)
Table 5 r5 c2: '0.670 ± 0.029' → '0.689 ± 0.011' (bold=False)
Table 5 r5 c3: '0.649 ± 0.076' → '0.703 ± 0.173' (bold=False)
P68: 'The rule chosen the weighted sampler during validation. On the test set the default square-rooted weights achieved a macro-F1 (95% confidence interval 0.641–0.738 versus 0.615–0.712 for the sampler) yet the intervals overlap. Inverse-frequency weights produced the melanoma recall but at the expense of accuracy. A single seed can mislead. Our original single-seed EfficientNet-B0, evaluated on views reached a macro-F1 of 0.721 which is higher, than the mean macro-F1 of the same recipe when using three seeds.' → 'The rule chose focal loss on validation (macro-F1 0.688, within 0.005 of no weighting, with the higher melanoma recall). On the test set the weighted sampler achieved the highest macro-F1 (95% confidence interval 0.635–0.727 versus 0.619–0.718 for focal loss) yet the intervals overlap. The weighted sampler produced the highest melanoma recall but at the expense of accuracy. A single seed can mislead: our single-seed EfficientNet-B0, evaluated on three views, reached a macro-F1 of 0.708, higher than the three-seed mean of the same recipe (0.679).'
Table 6 r1 c1: '−0.014 [−0.051, +0.023]' → '−0.008 [−0.040, +0.024]'
Table 6 r1 c2: '−0.014 [−0.045, +0.017]' → '−0.036 [−0.074, +0.005]'
Table 6 r1 c3: '+0.015 [+0.002, +0.030]' → '+0.017 [+0.007, +0.027]'
Table 6 r2 c1: '+0.058 [+0.022, +0.096]' → '+0.049 [+0.012, +0.088]'
Table 6 r2 c2: '−0.048 [−0.089, −0.009]' → '+0.108 [+0.065, +0.152]'
Table 6 r2 c3: '+0.061 [+0.046, +0.078]' → '−0.029 [−0.043, −0.015]'
P72: '(weighted sampler)' → '(focal loss)'
P73: 'Table IV Skin-tone augmentation shows no effect on HAM10000.' → 'Table IV: skin-tone augmentation did not help on HAM10000; it lowered melanoma recall (−0.094, 95% CI −0.148 to −0.038) and slightly raised specificity.'
P73: 'The U-Net reached a test Dice score of 0.943 ± 0.001 while Otsu thresholding gave 0.652. Training on crops raised macro-F1 and specificity. Training on crops lowered malignant sensitivity. Both confidence intervals exclude zero cropping trades missed cancers, for fewer false alarms.' → 'The U-Net reached a test Dice score of 0.949 ± 0.001 while Otsu thresholding gave 0.691. Training on crops raised macro-F1 and malignant sensitivity and lowered specificity, all with confidence intervals that exclude zero: cropping trades some false alarms for fewer missed cancers.'
P7: 'test macro-F1 of 0.747' → 'test macro-F1 of 0.771'
P7: 'cropping lesions increased macro-F1 while decreasing sensitivity for malignant lesions' → 'cropping lesions increased both macro-F1 and sensitivity for malignant lesions'
P43: 'All experiments are implemented in PyTorch [29]. Run on an Apple M2 Pro GPU.' → 'All experiments are implemented in PyTorch [29] and run on NVIDIA Tesla T4 GPUs (Kaggle).'
P81: 'macro-F1 score of 0.747' → 'macro-F1 score of 0.771'
P13: 'for -tumour conditions' → 'for non-tumour conditions'
P45: 'percentage of -neoplastic images' → 'percentage of non-neoplastic images'
P25: 'Melanoma (mel) cell carcinoma (bcc)' → 'Melanoma (mel), basal cell carcinoma (bcc)'
P28: 'All of those five images are cell carcinomas.' → 'All of those five images are basal cell carcinomas.'
P64: 'the ensemble was 12.5% of the time' → 'the ensemble was correct 12.5% of the time'
P66: 'does not help with cell carcinomas' → 'does not help with basal cell carcinomas'
P84: 'All of them are cell carcinomas' → 'All of them are basal cell carcinomas'