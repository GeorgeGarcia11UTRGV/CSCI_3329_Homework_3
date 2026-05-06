# CSCI 3329 — Homework 3 Report
## 1. Dataset
- Autistic Spectrum Disorder Screening Data for Children / [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children) / Samples: 292 / Clases: 2
- Class distribution (table or bar chart)
  
| Class | Count | Percentage |
|-------|-------|------------|
| No    | 151   | 51.7%      |
| Yes   | 141   | 48.3%      |

## 2. Preprocessing
- Missing-value handling
  - Converted '?' strings to NaN and dropped those rows.
- Encoding and scaling decisions, with rationale
  - Used LabelEncoder for binary and categorical features (Gender, Jaundice, etc.) to convert text into numerical format.
  - Used StandardScaler on numerical features (age and result). This ensures that distance-based models (KNN) and gradient-based models (MLP) treat all features with equal importance.
  - Removed age_desc (redundant), used_app_before, and relation as they do not contribute to the classification logic.

## 3. Part 2 — Algorithm Comparison
| Algorithm | Mean Accuracy | Std |
|-----------|---------------|-----|
| Linear Classifier    | 0.9061          | 0.1032
| Logistic Regression  | 0.9996          | 0.0040
| KNN                  | 0.8572          | 0.0702
| Gaussian NB          | 0.9672          | 0.0380
| Neural Network       | 0.9871          | 0.0231

## 4. Part 3 — Feature Selection
- Search method and justification
   - Forward Selection with 20 features, exhaustive search is computationally expensive. Forward selection provides a greedy heuristic that finds an optimal subset in $O(m^2)$ time. 
  
| Algorithm | Best Feature Subset | Mean Accuracy | Std |
|-----------|--------------------|---------------|-----|
| Linear Classifier | 8 features: ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'result'] | 0.9980 | 0.0097 |
| Logistic Regression | 8 features: ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A6_Score', 'A7_Score', 'age', 'result'] | 1.0000 | 0.0013 |
| KNN | 8 features: ['A1_Score', 'A2_Score', 'A5_Score', 'A6_Score', 'A10_Score', 'gender', 'autism', 'result'] | 0.9733 | 0.0365 |
| Gaussian NB | 8 features: ['A2_Score', 'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'country_of_res', 'result'] | 0.9746 | 0.0359 |
| Neural Network | 8 features: ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'result'] | 1.0000 | 0.0000 |

## 5. Discussion
- Part 2 vs Part 3 comparison
  - Feature selection improved/maintained the performance of most models while reducing the complexity of the input data.
- Per-algorithm observations
  - Gaussian NB had an improvement because removing correlated features helps satisfy the algorithm's independence assumption.
  - KNN stability increased by removing noisy features that distorted distance calculations.
- Limitations and ideas for improvement
  - The small sample size (n=242) makes the models sensitive to specific outliers in the dataset.

## 6. Reproduction
- Python Version: 3.10+
- Key Libraries: pandas, scikit-learn, ucimlrepo
- Run Command: pip install ucimlrepo
