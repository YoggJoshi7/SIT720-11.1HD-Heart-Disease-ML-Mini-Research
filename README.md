# SIT720 11.1HD – Heart Disease ML Mini Research

This repository contains the implementation and supporting materials for the **SIT720 11.1HD Mini Research** assessment at **Deakin University**.

The research investigates machine-learning approaches for early heart disease prediction through:

1. Reproduction of selected experiments from a published research paper.
2. Investigation of duplicate observations and their effect on evaluation.
3. Development of a proposed group-aware Support Vector Machine (SVM) pipeline.
4. Comparison of the proposed approach with controlled group-aware baseline models.

---
Installations: 
git clone https://github.com/YoggJoshi7/SIT720-11.1HD-Heart-Disease-ML-Mini-Research.git
cd SIT720-11.1HD-Heart-Disease-ML-Mini-Research
Requirements: 
pip install -r requirements.txt
Launch Notebook:
jupyter notebook
Notebook Name:
SIT720 11.1HD ML Mini Research.ipynb


## Reference Paper

Bhagat, P., Sharma, A., & Agarwal, S. (2025).

**An efficient stacking-based ensemble technique for early heart attack prediction.**

*Multimedia Tools and Applications.*

DOI: `10.1007/s11042-024-19293-7`

---

## Dataset

The experiments use the supplied `heart.csv` dataset.

| Property | Value |
|---|---:|
| Observations | 1,025 |
| Predictors | 13 |
| Target | Binary classification |
| Missing values | 0 |
| Class 0 | 499 |
| Class 1 | 526 |

---

# Part 1 – Reproduction

Six individual machine-learning models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gaussian Naive Bayes
- K-Nearest Neighbours

A stacking-based ensemble was also evaluated.

### Initial Reproduction Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8098 | 0.7619 | 0.9143 | 0.8312 | 0.9298 |
| Decision Tree | 0.9854 | 1.0000 | 0.9714 | 0.9855 | 0.9857 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.8293 | 0.8070 | 0.8762 | 0.8402 | 0.9043 |
| KNN | 0.8634 | 0.8738 | 0.8571 | 0.8654 | 0.9629 |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

# Duplicate-Data Investigation

The dataset was examined for duplicated predictor patterns.

The analysis identified:

- **723 duplicated observations/predictor patterns**
- **302 unique predictor patterns**
- **202 of 205 initial test observations** had predictor patterns also present in the training data
- **98.54% test-to-training predictor-pattern overlap**
- No duplicate predictor groups with conflicting labels were identified.

This motivated the use of group-aware splitting for the second stage.

A deduplicated sensitivity analysis of the stacking approach produced:

| Metric | Score |
|---|---:|
| Accuracy | 0.7541 |
| Precision | 0.7647 |
| Recall | 0.7879 |
| F1 | 0.7761 |
| ROC-AUC | 0.8680 |

---

# Part 2 – Proposed Group-Aware SVM

The proposed pipeline combines:

**Mutual Information → StandardScaler → RBF SVM**

A group-aware evaluation strategy was used so that identical predictor patterns were not shared between training and testing partitions.

### Group-Aware Split

| | Observations | Groups |
|---|---:|---:|
| Training | 821 | 242 |
| Testing | 204 | 60 |
| Shared groups | 0 | 0 |

---

## Hyperparameter Optimisation

A grid search using `StratifiedGroupKFold` cross-validation was performed.

### Search space

**Number of selected features:**

`5, 7, 9, 11, 13`

**C:**

`0.1, 1, 10, 100`

**Gamma:**

`scale, auto, 0.01, 0.1`

The optimisation objective was ROC-AUC.

### Selected Configuration

| Parameter | Value |
|---|---|
| Feature selection | Mutual Information |
| Number of features | 11 |
| Kernel | RBF |
| C | 100 |
| Gamma | `scale` |
| Group-aware CV AUC | 0.8433 |

### Selected Features

```text
age
sex
cp
trestbps
chol
thalach
exang
oldpeak
slope
ca
thal
```
Reproducibility Notes:

The implementation uses explicitly defined train-test and group-aware evaluation procedures.
The notebook contains:
- Data loading and preprocessing
- Individual model evaluation
- Stacking reproduction
- Duplicate-pattern analysis
- Group-aware data splitting
- Mutual Information feature selection
- SVM hyperparameter optimisation
- Group-aware baseline comparison
- Nested group-aware cross-validation
- Performance visualisation
The exact experimental details of the reference paper are not fully specified in all areas. Therefore, undocumented implementation details are not presented as exact reproductions.

Limitations
- The dataset contains substantial duplication in predictor patterns.
- The exact implementation details of the reference paper are not completely documented.
- Results are specific to the supplied dataset and experimental protocol.
- The proposed model is evaluated as a machine-learning research experiment and is not a clinically validated diagnostic system.

Author
Yogg Kunal Joshi
Master of Applied Artificial Intelligence (Professional)
Deakin University, Australia
SIT720 – Machine Learning

