# Traffic Accidents Prediction and Risk Analysis

**Traffic Accidents Prediction and Risk Analysis** is an end-to-end project that builds a pipeline to predict whether a traffic accident will occur and to produce actionable risk insights from historical accident records. The repository contains data cleaning, exploratory data analysis (EDA), feature engineering, model training (Logistic Regression, Random Forest, XGBoost, Gradient Boosting), hyperparameter tuning, evaluation, and basic risk-profiling.

---

## 🔍 Project Overview

**Goal:** predict the binary target `Accident` (0 = no accident, 1 = accident) and identify high-risk scenarios to help transportation authorities prioritize safety interventions.

**Main steps implemented**
- Data ingestion, validation and missing-value handling
- Exploratory Data Analysis
- Feature engineering
- One-hot encoding of categorical variables and robust scaling
- Model training and evaluation
- Hyperparameter tuning
- Feature importance reporting and simple risk-profiling

---

## Dataset (summary)

- **Original size:** 840 rows × 14 columns
- **Key fields:** `Weather`, `Road_Type`, `Time_of_Day`, `Traffic_Density`, `Speed_Limit`, `Number_of_Vehicles`, `Driver_Alcohol`, `Accident_Severity`, `Road_Condition`, `Vehicle_Type`, `Driver_Age`, `Driver_Experience`, `Road_Light_Condition`, `Accident` (target)
- **Download Link:** https://www.kaggle.com/datasets/denkuznetz/traffic-accident-prediction?resource=download
  
> The dataset file used in this analysis is `dataset_traffic_accident.csv`.

---

## Key Steps / Pipeline

1. **Data ingestion & validation**
   - Load CSV into a Pandas DataFrame.
   - Inspect dtypes, non-null counts, and target distribution.

2. **Missing value handling**
   - Drop rows with missing target `Accident`.
   - Numeric features: imputed with **median**.
   - Categorical features: imputed with **mode**.

3. **Exploratory Data Analysis (EDA)**
   - Histograms (with mean/median lines), boxplots, pairwise scatter matrix (scatter_matrix), correlation heatmap.
   - Categorical analysis (accident probability by weather, time of day, road type, alcohol involvement).

4. **Feature Engineering**
   - `Risk_Score` — heuristic weighted combination of Traffic_Density, normalized Speed_Limit, Driver_Alcohol, and inverse of Driver_Experience.
   - Binned categorical features: `Age_Group`, `Experience_Group`, `Speed_Category`, `Time_Category`.

5. **Preprocessing**
   - One-hot encode categorical variables (`pd.get_dummies(..., drop_first=True)`).
   - Scale numeric columns with `RobustScaler` (robust to outliers).
   - Stratified train/test split (80/20) to preserve class balance.

6. **Modeling & Evaluation**
   - Models trained: **Logistic Regression**, **Random Forest**, **XGBoost**, **Gradient Boosting**.
   - Cross-validation (StratifiedKFold, 5 folds) using **F1-score**.
   - Test metrics: accuracy, precision, recall, F1, ROC-AUC; confusion matrix plots.

7. **Hyperparameter Tuning**
   - Performed `GridSearchCV` (3-fold CV, scoring=`f1`) for the best model (selected by test F1). In this run, **XGBoost** was tuned.

8. **Feature Importance & Risk Analysis**
   - Extracted feature importances (tree-based models).
   - Flagged “high-risk” test cases using a probability threshold (0.7) and reported summary statistics for those cases.

---

## Results (from the analysis run)

- **Feature matrix shape after encoding:** (798, 36)  
- **Train / Test split:** 638 train / 160 test  
- **Cross-Validation (F1) mean (approx):**
  - Logistic Regression: **0.0562**
  - Random Forest: **0.1630**
  - XGBoost: **0.2499**
  - Gradient Boosting: **0.2371**

- **Best model selected for tuning:** **XGBoost**  

- **Tuned XGBoost — Test performance:**
  - **F1 Score:** `0.1972`
  - **ROC AUC:** `0.4794`
  - **Accuracy:** `0.6438`
  - **Precision:** `0.3043`
  - **Recall:** `0.1458`

- **High-risk cases (probability > 0.7):** **5.62%** of test set  
  - Characteristics of high-risk subset (averages from this run):  
    - Average Traffic Density: 0.17 (scaled)  
    - Average Speed Limit: 0.44 (scaled)  
    - Alcohol involvement: 22.22%  
    - Average Risk Score: 0.39

- **Top contributing features (example top 10 from feature importances):**
  - `Road_Type_Mountain Road`, `Traffic_Density`, `Vehicle_Type_Truck`, `Time_of_Day_Night`, `Age_Group_Adult`, `Weather_Foggy`, `Speed_Category_Medium`, `Road_Light_Condition_Daylight`, `Road_Type_Rural Road`, `Time_of_Day_Evening`.
