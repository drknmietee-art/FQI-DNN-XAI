# FQI-DNN-XAI: A Fuzzy Quantum-Inspired Deep Neural Network with Explainable AI for Ischemic Heart Disease Prediction

## Overview

This repository contains the complete MATLAB implementation of the **FQI-DNN-XAI** framework, a novel hybrid model that integrates:

1. **Takagi-Sugeno Fuzzy Inference System** for linguistic feature encoding and uncertainty modeling
2. **Quantum-Inspired Parameterized Layers** with rotation gates and entanglement-mimicking operations
3. **Deep Neural Network with Attention Mechanism** for hierarchical feature extraction
4. **Explainable AI (SHAP + LIME)** for clinical interpretability

## Requirements

- **MATLAB R2021b** or later
- **Deep Learning Toolbox**
- **Fuzzy Logic Toolbox**
- **Statistics and Machine Learning Toolbox**

## Dataset

The framework uses the **Cleveland Heart Disease** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease):
- 303 instances, 13 clinical features, binary target (IHD presence/absence)
- The dataset is automatically downloaded when you run the code

## Project Structure

```
FQI-DNN-XAI/
├── main_FQI_DNN_XAI.m              # Main script - run this
├── fuzzyMembershipEncoding.m        # Gaussian fuzzy membership (Eqs. 2-4)
├── computeFiringStrengths.m         # Takagi-Sugeno rule firing (Eqs. 5-7)
├── quantumInspiredTransform.m       # Quantum-inspired layers (Eqs. 8-13)
├── buildAndTrainFQIDNN.m            # DNN with attention (Eqs. 14-17)
├── predictFQIDNN.m                  # Model prediction
├── computeMetrics.m                 # Performance metrics (Eqs. 21-24)
├── applySMOTE.m                     # SMOTE oversampling
├── computeKernelSHAP.m              # Kernel SHAP explainability (Eq. 18)
├── runBaselineComparisons.m         # Baseline classifiers
├── generateAllFigures.m             # All 12 paper figures
├── data/                            # Dataset (auto-downloaded)
│   └── cleveland.csv
├── results/                         # Generated results
│   ├── Figure2_MembershipFunctions.png
│   ├── Figure3_FuzzyRuleActivation.png
│   ├── Figure5_SHAPAnalysis.png
│   ├── Figure6_LIMEExplanations.png
│   ├── Figure7_ROCCurves.png
│   ├── Figure8_ConfusionMatrices.png
│   ├── Figure9_AblationStudy.png
│   ├── Figure10_FeatureInteractions.png
│   ├── Figure11_Convergence.png
│   ├── Figure12_StatisticalAnalysis.png
│   ├── Table4_PerformanceComparison.csv
│   ├── Table6_AblationStudy.csv
│   └── FQI_DNN_XAI_Results.mat
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/FQI-DNN-XAI.git
   cd FQI-DNN-XAI
   ```

2. Open MATLAB and navigate to the project folder.

3. Run the main script:
   ```matlab
   main_FQI_DNN_XAI
   ```

4. The script will:
   - Download the Cleveland dataset (if not present)
   - Preprocess data (normalization, missing value handling)
   - Apply fuzzy membership encoding
   - Apply quantum-inspired feature transformation
   - Train the FQI-DNN-XAI model with 10-fold cross-validation
   - Run ablation studies
   - Compare with 6 baseline classifiers
   - Generate all 12 figures and result tables
   - Save everything to the `results/` folder

## Key Results

| Method | Accuracy (%) | Sensitivity (%) | F1-Score (%) | AUC-ROC |
|--------|-------------|-----------------|-------------|---------|
| Logistic Regression | ~83 | ~82 | ~83 | ~0.90 |
| SVM (RBF) | ~85 | ~84 | ~84 | ~0.91 |
| Random Forest | ~87 | ~86 | ~86 | ~0.92 |
| Gradient Boosting | ~88 | ~87 | ~88 | ~0.94 |
| **FQI-DNN-XAI (Proposed)** | **~97** | **~98** | **~97** | **~0.99** |

## Mathematical Framework

The paper equations are implemented as follows:

| Equation | Description | File |
|----------|-------------|------|
| Eq. 1 | Min-Max Normalization | `main_FQI_DNN_XAI.m` |
| Eqs. 2-4 | Gaussian Membership Functions | `fuzzyMembershipEncoding.m` |
| Eqs. 5-7 | Takagi-Sugeno Firing Strengths | `computeFiringStrengths.m` |
| Eqs. 8-13 | Quantum-Inspired Transform | `quantumInspiredTransform.m` |
| Eqs. 14-17 | DNN with Attention & Loss | `buildAndTrainFQIDNN.m` |
| Eq. 18 | SHAP Values | `computeKernelSHAP.m` |
| Eqs. 21-24 | Evaluation Metrics | `computeMetrics.m` |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2026fqidnnxai,
  title={A Fuzzy Quantum-Inspired Deep Neural Network with Explainable AI for Ischemic Heart Disease Prediction},
  author={Author 1 and Author 2 and Author 3},
  journal={Journal Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Cleveland Heart Disease dataset: Robert Detrano, M.D., Ph.D., V.A. Medical Center, Long Beach and Cleveland Clinic Foundation
- UCI Machine Learning Repository
