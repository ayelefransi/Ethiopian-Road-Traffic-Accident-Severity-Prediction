# Ethiopian Road Traffic Accident Severity Prediction
## Project Documentation

---

## Table of Contents
1. Project Overview
2. Dataset
3. Problem Statement
4. ML Pipeline
5. Models and Regularization
6. Performance Results
7. Project Structure
8. How to Run
9. API Reference
10. Key Findings

---

## 1. Project Overview

This project builds an end-to-end machine learning system to predict the severity of road traffic accidents in Addis Ababa, Ethiopia.

The system classifies each accident into one of three severity levels:
- Slight Injury
- Serious Injury
- Fatal injury

It is designed for use by the Addis Ababa Traffic Police and road safety authorities to understand risk patterns and allocate emergency resources more effectively.

---

## 2. Dataset

| Property | Detail |
|---|---|
| Source 1 | Mendeley Data (Addis Ababa Sub-city Police Depts, 2017-2020) |
| Source 2 | Figshare (Addis Ababa Police Depts, 2016-2022) |
| Total records | 25,380 |
| Features | 24 |
| Target | Accident_severity (3 classes) |
| DOI (Mendeley) | 10.17632/xytv86278f.2 |

### Feature Categories

Time Features: Time of Day, Day of Week, Year

Driver Features: Age Band, Sex, Education Level, Driving Experience, Vehicle-Driver Relation

Road Features: Lane/Median Type, Junction Type, Road Surface Type, Road Surface Condition, Road Alignment

Environment Features: Light Conditions, Weather Conditions

Accident Features: Type of Collision, Number of Vehicles, Number of Casualties, Vehicle Type, Cause of Accident, Pedestrian Movement, Vehicle Movement

Location Features: Area, Sub-district

### Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Slight Injury | 20,011 | 78.8% |
| Serious Injury | 5,104 | 20.1% |
| Fatal injury | 265 | 1.1% |

This is a heavily imbalanced dataset. Fatal injury accounts for only 1.1% of records.

---

## 3. Problem Statement

Road traffic accidents are one of the leading causes of death in Ethiopia. Addis Ababa accounts for a disproportionate number of accident fatalities. Predicting accident severity before or immediately after an incident allows:
- Faster emergency dispatch
- Better resource planning
- Policy targeting for high-risk zones and behaviors

---

## 4. ML Pipeline

```
Raw Data (CSV)
     |
     V
Drop metadata columns (Source, Year)
     |
     V
Encode target (label map: 0/1/2)
     |
     V
Train/Test Split (80/20, Stratified)
     |
     V
ColumnTransformer
  |-- Categorical: SimpleImputer (mode) -> OrdinalEncoder
  |-- Numerical:   SimpleImputer (median) -> StandardScaler
     |
     V
SMOTE (oversample minority classes in training only)
     |
     V
Classifier (with regularization)
     |
     V
Stratified 5-Fold Cross-Validation
     |
     V
Select best model by CV F1 (weighted)
     |
     V
Evaluate on held-out test set
     |
     V
Save model + metadata (joblib)
     |
     V
Deploy via Streamlit (frontend) + FastAPI (backend)
```

---

## 5. Models and Regularization

### Logistic Regression with L2 (Ridge)

- Regularization parameter C = 0.1 (strong L2 penalty)
- Lower C = stronger regularization = smaller coefficients
- Prevents overfitting when categorical features are one-hot encoded into wide feature space

### Random Forest (Structural Regularization)

- max_depth = 12 (limits tree complexity)
- min_samples_leaf = 5 (minimum samples per leaf)
- min_samples_split = 10 (minimum samples to make a split)
- max_features = sqrt (column subsampling per split)
- All four parameters work together to prevent trees from memorizing training data

### XGBoost with L1 + L2

- reg_alpha = 0.1 (L1 regularization on leaf weights)
- reg_lambda = 1.5 (L2 regularization on leaf weights)
- learning_rate = 0.1 (shrinkage: scales each tree contribution)
- subsample = 0.8 (row subsampling per tree)
- colsample_bytree = 0.8 (column subsampling per tree)
- max_depth = 6 (depth cap)

This combination of L1, L2, shrinkage and subsampling gives XGBoost strong protection against overfitting on imbalanced data.

### Why SMOTE?

Fatal injury has only 265 records. Without resampling, classifiers ignore the minority class and predict Slight Injury for almost every record. SMOTE creates synthetic minority class samples by interpolating between real minority samples in feature space. It is applied only on the training set to prevent data leakage.

---

## 6. Performance Results

### Cross-Validation Summary

| Model | Train Acc | CV Acc | CV F1 | Overfit Gap |
|---|---|---|---|---|
| Logistic Regression (L2) | 0.6188 | 0.6147 | 0.6725 | 0.0042 |
| Random Forest (Regularized) | 0.9285 | 0.8429 | 0.8249 | 0.0856 |
| XGBoost (L1+L2) | 0.9952 | 0.9316 | 0.9258 | 0.0636 |

### Final Test Set Results (XGBoost)

| Metric | Score |
|---|---|
| Accuracy | 93.28% |
| Precision (weighted) | 0.9273 |
| Recall (weighted) | 0.9328 |
| F1 Score (weighted) | 0.9272 |
| F1 Score (macro) | 0.6389 |
| ROC-AUC (OvR) | 0.9863 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Slight Injury | 0.94 | 0.99 | 0.96 | 4,002 |
| Serious Injury | 0.89 | 0.76 | 0.82 | 1,021 |
| Fatal injury | 0.50 | 0.08 | 0.13 | 53 |

Fatal injury remains the hardest class to predict due to extreme scarcity (53 test records).

---

## 7. Project Structure

```
rta_project/
|
|-- data/
|   |-- RTA_combined.csv         (merged Mendeley + Figshare dataset)
|
|-- notebooks/
|   |-- Ethiopia_RTA_Severity_Prediction.ipynb    (full ML pipeline)
|
|-- models/
|   |-- best_rta_model.pkl       (trained XGBoost pipeline)
|   |-- model_metadata.json      (metrics, features, class info)
|
|-- app/
|   |-- streamlit_app.py         (frontend web app)
|   |-- main.py                  (FastAPI backend)
|
|-- docs/
|   |-- README.md                (this file)
|   |-- 01_target_distribution.png
|   |-- 02_temporal_analysis.png
|   |-- 03_causes_severity.png
|   |-- 04_conditions_analysis.png
|   |-- 05_cv_comparison.png
|   |-- 06_confusion_matrix.png
|   |-- 07_roc_curves.png
|   |-- 08_per_class_metrics.png
|   |-- 09_feature_importance.png
|   |-- 10_regularization_analysis.png
```

---

## 8. How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn \
            joblib matplotlib seaborn streamlit fastapi uvicorn \
            plotly jupyter nbformat
```

### Run Jupyter Notebook

```bash
cd rta_project/notebooks
jupyter notebook Ethiopia_RTA_Severity_Prediction.ipynb
```

### Run Streamlit Frontend

```bash
cd rta_project
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Run FastAPI Backend

```bash
cd rta_project/app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the Swagger UI.

---

## 9. API Reference

### POST /predict

Predict severity for a single accident.

Request body example:
```json
{
  "Time": "Night (22-6)",
  "Day_of_week": "Friday",
  "Age_band_of_driver": "18-30",
  "Sex_of_driver": "Male",
  "Educational_level": "High school",
  "Vehicle_driver_relation": "Employee",
  "Driving_experience": "Below 1yr",
  "Lanes_or_Medians": "Undivided Two way",
  "Types_of_Junction": "Y Shape",
  "Road_surface_type": "Asphalt roads",
  "Road_surface_conditions": "Wet or damp",
  "Light_conditions": "Darkness - no lighting",
  "Weather_conditions": "Raining",
  "Type_of_collision": "Rollover",
  "Number_of_vehicles_involved": 3,
  "Number_of_casualties": 4,
  "Vehicle_type": "Lorry (41-100Q)",
  "Cause_of_accident": "Drunk driving",
  "Pedestrian_movement": "Crossing from nearside",
  "Vehicle_movement": "Going straight",
  "Type_of_vehicle": "Long lorry",
  "Road_allignment": "Tangent road with flat terrain",
  "Area_accident_occured": "Residential areas",
  "Sub_district": "Bole"
}
```

Response:
```json
{
  "predicted_severity": "Fatal injury",
  "severity_code": 2,
  "confidence": 0.874,
  "confidence_pct": "87.4%",
  "probabilities": {
    "Slight Injury": 0.041,
    "Serious Injury": 0.085,
    "Fatal injury": 0.874
  },
  "risk_level": "HIGH",
  "model_name": "XGBoost (L1+L2)",
  "model_accuracy": 0.9328
}
```

### GET /health

Returns model performance stats.

### GET /features

Returns all valid values for each categorical feature.

### POST /predict/batch

Accepts up to 100 records in a list. Returns predictions for all.

---

## 10. Key Findings

1. Night driving (22:00 to 06:00) and darkness with no street lighting are the strongest contributors to fatal accidents.

2. Drunk driving and overspeed are the top two causes with the highest fatal rates.

3. Rollover collisions have a disproportionately high fatality ratio compared to vehicle-to-vehicle collisions.

4. Roads outside Addis Ababa and escarpment road alignments produce more fatal outcomes.

5. Drivers with below 1 year of experience or no licence are significantly over-represented in serious and fatal accidents.

6. Stacking ensemble (CART + Extra Trees + Bagging, meta-learner: Logistic Regression) achieved 90.82% accuracy and 0.9822 ROC-AUC using only algorithms from the allowed list.

7. Fatal injury prediction remains the hardest class (F1 = 0.13) due to extreme class imbalance (265 records vs 20,011). Advanced techniques like cost-sensitive learning or focal loss would improve this further.

---

*Dataset licensed under CC BY 4.0. Mendeley Data, DOI: 10.17632/xytv86278f.2*
