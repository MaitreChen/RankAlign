# RankAlign-EEG

[English](README.md) | [简体中文](README_CN.md)

Official implementation of **RankAlign-EEG: Multi-Granularity EEG Emotion Recognition Based on Within-Subject Ranking Constraint and Cross-Subject Alignment**.

## Introduction

EEG emotion recognition is challenging because signals are noisy, labelled datasets are small, and physiological distributions vary substantially between subjects. A model trained on known participants can therefore learn subject-specific patterns and generalize poorly to unseen participants.

This project was developed for the **11th National College Student Biomedical Engineering Innovation Design Competition** (第十一届全国大学生生物医学工程创新设计竞赛), where it won the **First Prize**. The competition task is binary EEG emotion recognition involving healthy controls (HC) and participants with depression (DEP): determining whether an EEG trial represents a **neutral** or **positive** emotional state. The evaluation is cross-subject, meaning that test participants are not present during training.

The competition protocol provides an additional structural prior. Each test participant has eight trials—four positive and four neutral—in an unknown order. RankAlign-EEG uses this public constraint during inference: it ranks the eight positive-class probabilities within each participant and assigns the top four trials to the positive class instead of applying one global threshold.

RankAlign-EEG addresses three practical difficulties:

1. **Cross-subject distribution shift:** Euclidean Alignment (EA) [1] reduces subject-dependent covariance differences.
2. **Noisy and limited EEG observations:** multi-granularity modelling combines an EEGNet-style encoder [2] with differential entropy (DE) [3] and Welch spectral statistics [4].
3. **Subject-dependent probability calibration:** within-subject Top-4 ranking replaces a potentially biased global threshold.

## Dataset

The EEG signals contain 30 channels sampled at 250 Hz. One 10-second segment therefore has the shape `30 × 2500`.

### Training set

- 60 participants: 40 HC and 20 DEP.
- 40 ten-second EEG segments per participant.
- 20 neutral and 20 positive segments per participant.
- 2,400 segment-level samples in total.
- Every five consecutive segments of the same emotion form one training trial, producing eight trials per participant and 480 trial-level samples.

### Test sets

- Each test participant contains eight ten-second trials.
- The public protocol specifies four positive and four neutral trials per participant, while their order is unknown.
- Public and private test labels are not used for model training or fusion-weight fitting.

The competition data are not redistributed. Prepare the following layout:

```text
DATA_ROOT/
|-- train/
|   |-- HC/*timedata.mat
|   `-- DEP/*timedata.mat
`-- test/                   # optional for local OOF reproduction
    `-- P_test*.mat
```

Each training MAT file must contain `EEG_data_neu` and `EEG_data_pos`. Signals can be channel-first or channel-last, but must contain 30 channels.

## Method

The overall RankAlign-EEG architecture is shown below.

![RankAlign-EEG framework](figures/framework.svg)

*Figure 1. RankAlign-EEG combines cross-subject alignment, segment-level deep representation learning, trial-level spectral modelling, probability fusion, and within-subject Top-4 inference.*

```text
30-channel raw EEG
  |
  |-- Segment-level deep branch
  |     |-- subject-wise Euclidean Alignment [1]
  |     |-- EEGNetLite spatio-temporal encoder [2]
  |     |-- multi-fold / multi-seed probability ensemble
  |     `-- boundary-aware probability refinement
  |
  |-- Trial-level spectral branch A
  |     |-- DE [3] + absolute/relative Welch bandpower [4]
  |     |-- five-segment mean aggregation
  |     |-- subject-wise min-max normalization
  |     `-- Logistic Regression, C=0.05
  |
  |-- Trial-level spectral branch B
  |     |-- DE [3] + absolute/relative Welch bandpower [4]
  |     |-- five-segment mean aggregation
  |     |-- subject-wise z-score normalization
  |     `-- Logistic Regression, C=0.03
  |
  `-- Fixed probability fusion
        `-- within-subject Top-4 ranking -> final labels
```

### Euclidean Alignment

For subject \(s\), the mean covariance matrix is

$$
R_s=\frac{1}{n_s}\sum_{i:s_i=s}\mathrm{cov}(X_i),
$$

and each trial is whitened using

$$
\widetilde X_i=R_s^{-1/2}X_i.
$$

This is the Euclidean Alignment transform introduced by He and Wu [1]. It is an unsupervised preprocessing operation and uses no emotion labels.

### Multi-granularity modelling

The segment branch uses a compact EEGNet-style architecture [2] to learn temporal rhythms and cross-channel spatial patterns from `1 × 30 × 2500` EEG tensors. The trial branches extract SEED-style differential entropy [3] and absolute/relative Welch bandpower [4]. During training, features from five consecutive segments are averaged to form a stable trial representation.

### Probability fusion

The fixed fusion used for the reported local result is

$$
p_{final}=0.600p_{segment}+0.305p_{minmax}+0.095p_{zscore}.
$$

The weights are fixed after subject-disjoint OOF validation. Raw rather than rank-normalized probabilities are fused.

### Within-subject Top-4 inference

The eight fused positive probabilities of each test participant are sorted in descending order. The top four trials are predicted as positive and the other four as neutral:

$$
\hat y_{s,j}=\mathbb{1}\left[\mathrm{rank}_s(p_{s,j})\leq4\right].
$$

Top-4 is an inference constraint derived from the public competition protocol. It is not a ranking loss and does not use hidden test labels.

## Results

### Local subject-disjoint validation

All local results use subject-level GroupKFold OOF predictions. No participant occurs in both the training and validation portions of a fold.

| Method | Accuracy (%) | AUC (%) |
|---|---:|---:|
| DE + SVM [3] | 79.17 | 84.51 |
| GDDN [5] | 70.50 | 74.21 |
| DeepConvNet [6] | 70.25 | 71.91 |
| TCN [7] | 69.92 | 71.24 |
| EEGConformer [8] | 69.25 | 72.28 |
| CADA [9] | 71.50 | 74.92 |
| MV-SSTMA-DA [10] | 84.17 | 88.72 |
| **RankAlign-EEG** | **84.58** | **89.76** |

To remain consistent with the competition presentation, the table uses the heading “Accuracy”. More precisely, the local 84.58% value is **within-subject ranking balanced accuracy (Rank-BAcc)**, not ordinary accuracy obtained with a global 0.5 threshold. All values are results from this project's unified local protocol; references identify the originating methods, not the source of the numerical values. Recent domain-adaptation rows are local reproductions or method-inspired proxies and should not be interpreted as numbers quoted directly from the original papers.

### Official test results

| Evaluation set | Metric | Result |
|---|---|---:|
| Local subject-disjoint OOF | Rank-BAcc | **84.58%** |
| Local subject-disjoint OOF | AUC | **89.76%** |
| Official public test set | Accuracy | **75.0%** |
| Official private test set | Accuracy | **82.5%** |

### Error analysis

The following analysis examines the confusion pattern and the decision margins of correctly and incorrectly classified trials. Most errors occur near the within-subject Top-4 decision boundary, indicating boundary ambiguity rather than random model failure.

![RankAlign-EEG error analysis](figures/error_analysis.svg)

*Figure 2. Error and decision-boundary analysis of RankAlign-EEG.*

### Representation visualization

The feature visualization compares the trial-level distributions before and after RankAlign-EEG using t-SNE [11]. Davies–Bouldin Index (DBI) [12] and Silhouette Coefficient (SC) [13] are provided as descriptive cluster-quality measures. The aligned and fused representation produces more compact within-class structures and clearer separation between neutral and positive trials across HC and DEP groups.

![RankAlign-EEG representation visualization](figures/visualization_analysis.svg)

*Figure 3. Trial-level feature-distribution visualization before and after RankAlign-EEG.*

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Reproduction

Generate the two spectral OOF branches:

```bash
python source/train_spectral.py --data-root DATA_ROOT --output-dir outputs/spectral
```

Evaluate the fixed fusion after producing the segment OOF predictions:

```bash
python source/fuse_evaluate.py \
  --data-root DATA_ROOT \
  --segment-oof outputs/segment_oof.csv \
  --minmax-oof outputs/spectral/minmax_oof.csv \
  --zscore-oof outputs/spectral/zscore_oof.csv \
  --output outputs/rankalign_oof.csv
```

Run the lightweight tests:

```bash
set PYTHONPATH=source
python -m unittest discover -s source/tests
```

## Repository structure

```text
RankAlign/
|-- README.md
|-- README_EN.md
|-- README_CN.md
|-- requirements.txt
|-- configs/rankalign.json
|-- figures/
|   |-- framework.svg
|   |-- error_analysis.svg
|   `-- visualization_analysis.svg
`-- source/
    |-- rankalign/
    |-- train_spectral.py
    |-- fuse_evaluate.py
    `-- tests/
```

## References

1. H. He and D. Wu, “Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach,” *IEEE Transactions on Biomedical Engineering*, 67(2):399–410, 2020. [doi:10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914)
2. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain–Computer Interfaces,” *Journal of Neural Engineering*, 15(5):056013, 2018. [doi:10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
3. W.-L. Zheng and B.-L. Lu, “Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks,” *IEEE Transactions on Autonomous Mental Development*, 7(3):162–175, 2015. [doi:10.1109/TAMD.2015.2431497](https://doi.org/10.1109/TAMD.2015.2431497)
4. P. D. Welch, “The Use of Fast Fourier Transform for the Estimation of Power Spectra,” *IEEE Transactions on Audio and Electroacoustics*, 15(2):70–73, 1967. [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
5. B. Chen et al., “GDDN: Graph Domain Disentanglement Network for Generalizable EEG Emotion Recognition,” *IEEE Transactions on Affective Computing*, 2024. [doi:10.1109/TAFFC.2024.3371540](https://doi.org/10.1109/TAFFC.2024.3371540)
6. R. T. Schirrmeister et al., “Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization,” *Human Brain Mapping*, 38(11):5391–5420, 2017. [doi:10.1002/hbm.23730](https://doi.org/10.1002/hbm.23730)
7. T. M. Ingolfsson et al., “EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces,” in *2020 IEEE International Conference on Systems, Man, and Cybernetics*, 2020. [doi:10.1109/SMC42975.2020.9283028](https://doi.org/10.1109/SMC42975.2020.9283028)
8. Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization,” *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 31:710–719, 2023. [doi:10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
9. H. Huang, X. Si, Y. Han, and D. Ming, “A Novel Conditional Adversarial Domain Adaptation Network for EEG Cross-Subject Emotion Recognition,” *IEEE Transactions on Affective Computing*, 16(4):2905–2917, 2025. [doi:10.1109/TAFFC.2025.3588873](https://doi.org/10.1109/TAFFC.2025.3588873)
10. L. Zhang, H. Shi, Z. Li, W.-L. Zheng, and B.-L. Lu, “Multi-View Self-Supervised Domain Adaptation for EEG-Based Emotion Recognition,” *IEEE Transactions on Affective Computing*, 16(4):3055–3066, 2025. [doi:10.1109/TAFFC.2025.3574868](https://doi.org/10.1109/TAFFC.2025.3574868)
11. L. van der Maaten and G. Hinton, “Visualizing Data Using t-SNE,” *Journal of Machine Learning Research*, 9:2579–2605, 2008. [JMLR paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
12. D. L. Davies and D. W. Bouldin, “A Cluster Separation Measure,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2):224–227, 1979. [doi:10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909)
13. P. J. Rousseeuw, “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,” *Journal of Computational and Applied Mathematics*, 20:53–65, 1987. [doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

## License

License selection is pending. Do not redistribute the competition dataset unless its original terms explicitly permit redistribution.

## Conclusion

RankAlign-EEG tackles cross-subject EEG emotion recognition through a coordinated pipeline rather than a single oversized network. Euclidean Alignment reduces subject-dependent covariance shift, the segment and trial branches combine complementary spatio-temporal and spectral evidence, and Top-4 inference addresses subject-dependent probability calibration under the public competition protocol. The method achieved 84.58% local Rank-BAcc and 89.76% AUC, together with 75.0% accuracy on the official public test set and 82.5% on the official private test set. These results demonstrate the value of combining cross-subject alignment, multi-granularity evidence, and structured within-subject decisions.

## 📞 Contact

For any questions or suggestions about this project, everyone is welcome to raise **issues**!

Please also feel free to contact [hongbinchen@stu.njmu.edu.cn](mailto:hongbinchen@stu.njmu.edu.cn).
