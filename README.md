# Wine Quality Classifier

## Problem Statement

Predict whether a wine is **good** (quality score ≥ 7) or **not good** (quality < 7) based on its physicochemical properties. This is a binary classification task using 6 different ML models.

## Dataset

- **Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples**: 6497 (1599 red + 4898 white)
- **Features**: 12 (11 physicochemical properties + wine type)
- **Target**: Binary — 1 if quality ≥ 7 (good), 0 otherwise
- **Class split**: 1277 good, 5220 not good (imbalanced)

| # | Feature | Description |
|---|---------|-------------|
| 1 | fixed acidity | tartaric acid (g/dm³) |
| 2 | volatile acidity | acetic acid (g/dm³) |
| 3 | citric acid | (g/dm³) |
| 4 | residual sugar | (g/dm³) |
| 5 | chlorides | sodium chloride (g/dm³) |
| 6 | free sulfur dioxide | (mg/dm³) |
| 7 | total sulfur dioxide | (mg/dm³) |
| 8 | density | (g/cm³) |
| 9 | pH | acidity level |
| 10 | sulphates | potassium sulphate (g/dm³) |
| 11 | alcohol | (% vol) |
| 12 | type | 0 = red, 1 = white |

## Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8538 | 0.7749 | 0.6250 | 0.6445 | 0.6346 | 0.5434 |
| KNN | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| Naive Bayes | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Random Forest (Ensemble) | 0.8923 | 0.9120 | 0.8333 | 0.5664 | 0.6744 | 0.6291 |
| XGBoost (Ensemble) | 0.8792 | 0.9021 | 0.7281 | 0.6172 | 0.6681 | 0.5979 |

## Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | High accuracy but very low recall (0.26) — it barely identifies good wines. The linear decision boundary struggles with this imbalanced dataset. High AUC (0.80) suggests it ranks probabilities reasonably but the default threshold is poor for the minority class. |
| Decision Tree | Balanced precision-recall tradeoff (0.63/0.64) with decent F1. Tends to overfit on training data, which shows in the lower AUC (0.77) compared to ensemble methods. Still, it captures non-linear patterns that logistic regression misses. |
| KNN | Moderate performance across all metrics. AUC (0.83) is better than Decision Tree, meaning it ranks predictions well, but the hard classification at k=5 loses some of that. Sensitive to the feature scaling we applied. |
| Naive Bayes | Lowest accuracy (0.73) due to many false positives — it predicts "good" too often. The independence assumption doesn't hold well since wine features are correlated (e.g., density and alcohol). However, it has the second-highest recall (0.62), catching more actual good wines. |
| Random Forest (Ensemble) | Best overall model — highest accuracy (0.89), AUC (0.91), precision (0.83), and MCC (0.63). Bagging reduces the overfitting seen in single Decision Trees. The precision-recall gap suggests it's conservative in predicting "good", which keeps false positives low. |
| XGBoost (Ensemble) | Second best model, very close to Random Forest. Slightly lower precision but similar recall and AUC (0.90). Boosting focuses on hard-to-classify samples, giving it strong performance on the minority class. The MCC (0.60) confirms solid performance even with class imbalance. |

## How to Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Live App

[Streamlit App](https://wine-classifier.streamlit.app)
