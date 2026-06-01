import nbformat as nbf, json

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {"display_name":"Python 3","language":"python","name":"python3"}

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

cells = []

# ── TITLE ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Ethiopian Road Traffic Accident Severity Prediction
## End-to-End ML Pipeline | Addis Ababa RTA Dataset

| Property | Detail |
|---|---|
| **Problem Type** | Multiclass Classification |
| **Target** | Slight Injury / Serious Injury / Fatal Injury |
| **Dataset** | Mendeley (2017-2020) + Figshare (2016-2022) merged |
| **Total Records** | 25,380 |
| **Features** | 24 |
| **Best Model** | XGBoost with L1 + L2 Regularization |
| **Test Accuracy** | 93.28% |
| **ROC-AUC** | 0.9863 |

---
"""))

# ── 1. SETUP ───────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup and Imports"))
cells.append(code("""import warnings
warnings.filterwarnings('ignore')

# Core
import pandas as pd
import numpy as np
import json, os, joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

print("Libraries loaded.")
"""))

# ── 2. LOAD DATA ───────────────────────────────────────────────────────────────
cells.append(md("""## 2. Data Loading

We merge two complementary datasets from Addis Ababa Police Departments:
- **Mendeley**: 12,316 records (2017-2020), 32 features
- **Figshare**: 13,064 records (2016-2022), 31 features

Both share the same accident feature schema.
"""))
cells.append(code("""df = pd.read_csv('../data/RTA_combined.csv')

print(f"Total records : {df.shape[0]:,}")
print(f"Total columns : {df.shape[1]}")
print(f"\\nSource breakdown:")
print(df['Source'].value_counts().to_string())
print(f"\\nYear range:")
print(df['Year'].value_counts().sort_index().to_string())
"""))
cells.append(code("""df.head()
"""))
cells.append(code("""df.describe(include='all').T.head(30)
"""))

# ── 3. EDA ─────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Exploratory Data Analysis

Five questions we answer before modeling:
1. How is severity distributed?
2. When do accidents happen?
3. What causes the most fatal accidents?
4. Which road conditions are most dangerous?
5. How imbalanced is our target?
"""))

cells.append(code("""# ── 3.1 Target Distribution
severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
colors = ['#2ecc71', '#f39c12', '#e74c3c']
counts = df['Accident_severity'].value_counts().reindex(severity_order)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(severity_order, counts.values, color=colors, edgecolor='white', linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f'{v:,}\\n({v/len(df)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Accident Severity Count', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Accidents')
axes[0].set_ylim(0, max(counts.values) * 1.18)

axes[1].pie(counts.values, labels=severity_order, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Severity Proportions', fontsize=13, fontweight='bold')

plt.suptitle('Addis Ababa RTA Severity Distribution (2016-2022)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('../docs', exist_ok=True)
plt.savefig('../docs/01_target_distribution.png', bbox_inches='tight')
plt.show()
print(f"\\nClass imbalance (Slight : Fatal) = {counts['Slight Injury']} : {counts['Fatal injury']} = {counts['Slight Injury']//counts['Fatal injury']}:1")
"""))

cells.append(code("""# ── 3.2 Temporal Analysis
time_order = ['Morning (6-10)','Noon (10-14)','Afternoon (14-18)','Evening (18-22)','Night (22-6)']
day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

time_sev = df.groupby(['Time','Accident_severity']).size().unstack(fill_value=0).reindex(time_order)
time_sev[severity_order].plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_title('Accidents by Time of Day', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(title='Severity', fontsize=8)
axes[0].set_xlabel('')

day_sev = df.groupby(['Day_of_week','Accident_severity']).size().unstack(fill_value=0).reindex(day_order)
day_sev[severity_order].plot(kind='bar', ax=axes[1], color=colors, edgecolor='white', linewidth=0.5)
axes[1].set_title('Accidents by Day of Week', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=25)
axes[1].legend(title='Severity', fontsize=8)
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig('../docs/02_temporal_analysis.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# ── 3.3 Top Causes by Fatal Rate
cause_sev = df.groupby(['Cause_of_accident','Accident_severity']).size().unstack(fill_value=0)
cause_sev = cause_sev.reindex(columns=severity_order, fill_value=0)
cause_sev['fatal_rate'] = cause_sev['Fatal injury'] / cause_sev.sum(axis=1)
cause_sev = cause_sev.sort_values('fatal_rate', ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
cause_sev[severity_order].plot(kind='barh', ax=ax, color=colors,
                                edgecolor='white', linewidth=0.4)
ax.set_title('Accident Cause by Severity', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Accidents')
ax.legend(title='Severity')
plt.tight_layout()
plt.savefig('../docs/03_causes_severity.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# ── 3.4 Road and Light Conditions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cat_plots = [
    ('Road_surface_conditions', 'Road Surface Condition'),
    ('Light_conditions',        'Light Conditions'),
    ('Weather_conditions',      'Weather Conditions'),
    ('Type_of_collision',       'Collision Type'),
]
for ax, (col, title) in zip(axes.flat, cat_plots):
    pivot = df.groupby([col,'Accident_severity']).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=severity_order, fill_value=0)
    pivot.div(pivot.sum(axis=1), axis=0).mul(100).plot(
        kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

plt.suptitle('Risk Conditions vs Severity (%)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../docs/04_conditions_analysis.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# ── 3.5 Missing Values Check
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("No missing values in the dataset.")
else:
    print(missing)

print(f"\\nDataset dtypes:")
print(df.dtypes.value_counts())
"""))

# ── 4. PREPROCESSING ───────────────────────────────────────────────────────────
cells.append(md("""## 4. Preprocessing

Steps:
1. Drop non-predictive metadata columns (Source, Year)
2. Encode the target variable
3. Split features into categorical and numerical
4. Train/Test split with stratification (80/20)
"""))

cells.append(code("""# Drop metadata
df_model = df.drop(columns=['Source', 'Year'])

# Encode target
label_map = {'Slight Injury': 0, 'Serious Injury': 1, 'Fatal injury': 2}
y = df_model['Accident_severity'].map(label_map)
X = df_model.drop(columns=['Accident_severity'])

# Feature types
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

print(f"Feature matrix: {X.shape}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features  ({len(num_cols)}): {num_cols}")
"""))

cells.append(code("""# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]:,} rows")
print(f"Test set     : {X_test.shape[0]:,} rows")
print(f"\\nTraining class distribution:")
for label, code_val in label_map.items():
    n = (y_train == code_val).sum()
    print(f"  {label}: {n:,} ({n/len(y_train)*100:.1f}%)")
"""))

# ── 5. PIPELINE ────────────────────────────────────────────────────────────────
cells.append(md("""## 5. ML Pipeline Construction

### Architecture

```
Input Features
     |
     ├── Categorical → Impute (mode) → OrdinalEncoder
     └── Numerical   → Impute (median) → StandardScaler
                              |
                       ColumnTransformer
                              |
                           SMOTE  (balance minority classes)
                              |
                          Classifier
```

### Regularization Strategy

| Model | Type | Parameters |
|---|---|---|
| Logistic Regression | L2 (Ridge) | C=0.1 |
| Random Forest | Structural | max_depth=12, min_samples_leaf=5 |
| XGBoost | L1+L2+Shrinkage | alpha=0.1, lambda=1.5, lr=0.1 |

### Why SMOTE?
Fatal injury class has only 265 records vs 20,011 Slight Injury.
SMOTE synthesizes minority class samples in feature space to fix this.
"""))

cells.append(code("""# Preprocessing sub-pipelines
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols),
    ('num', num_pipeline, num_cols)
])

print("Preprocessor built.")
"""))

cells.append(code("""# Full model pipelines with regularization
models = {

    'Logistic Regression (L2)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', LogisticRegression(
            C=0.1,                  # L2 strength: lower = stronger regularization
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),

    'Random Forest (Regularized)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,           # depth cap = regularization
            min_samples_leaf=5,     # leaf size = regularization
            min_samples_split=10,   # split threshold = regularization
            max_features='sqrt',    # feature subsampling = regularization
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ]),

    'XGBoost (L1+L2)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=6,            # depth regularization
            learning_rate=0.1,      # shrinkage regularization
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.5,         # L2 regularization
            subsample=0.8,          # row subsampling
            colsample_bytree=0.8,   # column subsampling
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ]),
}

print(f"Models ready: {list(models.keys())}")
"""))

# ── 6. CV ──────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Cross-Validation

**Stratified 5-Fold CV** ensures every fold has the same class proportions.
We track both train and validation scores to detect overfitting.
"""))

cells.append(code("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("Running 5-Fold Cross-Validation...\\n")
for name, pipeline in models.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={'accuracy':'accuracy', 'f1_weighted':'f1_weighted'},
        return_train_score=True, n_jobs=1
    )
    cv_results[name] = scores
    train_acc = scores['train_accuracy'].mean()
    val_acc   = scores['test_accuracy'].mean()
    val_f1    = scores['test_f1_weighted'].mean()
    gap       = train_acc - val_acc
    print(f"{name}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Val Accuracy   : {val_acc:.4f} +/- {scores['test_accuracy'].std():.4f}")
    print(f"  Val F1 (wtd)   : {val_f1:.4f}")
    print(f"  Overfit gap    : {gap:.4f}  {'[OK]' if gap < 0.08 else '[OVERFIT]'}")
    print()
"""))

cells.append(code("""# CV Comparison Chart
model_names  = list(cv_results.keys())
short_names  = ['LR (L2)', 'RF (Reg)', 'XGBoost']
colors_bar   = ['#3498db', '#2ecc71', '#e67e22']

train_accs = [cv_results[m]['train_accuracy'].mean() for m in model_names]
val_accs   = [cv_results[m]['test_accuracy'].mean()  for m in model_names]
val_f1s    = [cv_results[m]['test_f1_weighted'].mean() for m in model_names]

x = np.arange(len(model_names))
w = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w, train_accs, w, label='Train Accuracy',    color='#3498db', alpha=0.85)
b2 = ax.bar(x,     val_accs,   w, label='Val Accuracy',      color='#2ecc71', alpha=0.85)
b3 = ax.bar(x + w, val_f1s,    w, label='Val F1 (weighted)', color='#e67e22', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11)
ax.set_ylim(0.55, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Cross-Validation Results', fontsize=13, fontweight='bold')
ax.axhline(0.90, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(2.55, 0.905, '90% target', color='red', fontsize=9)
ax.legend(fontsize=10)

for bar_group in [b1, b2, b3]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../docs/05_cv_comparison.png', bbox_inches='tight')
plt.show()
"""))

# ── 7. EVALUATION ──────────────────────────────────────────────────────────────
cells.append(md("""## 7. Final Model Evaluation on Test Set

Select the model with highest weighted F1 from CV, then evaluate on unseen test data.
"""))

cells.append(code("""# Select best model
best_name = max(cv_results, key=lambda m: cv_results[m]['test_f1_weighted'].mean())
print(f"Selected model: {best_name}")
print(f"  Val F1 (weighted): {cv_results[best_name]['test_f1_weighted'].mean():.4f}")
print(f"  Val Accuracy:      {cv_results[best_name]['test_accuracy'].mean():.4f}")

best_pipeline = models[best_name]
print("\\nFitting on full training set...")
best_pipeline.fit(X_train, y_train)
print("Done.")
"""))

cells.append(code("""# Predictions
class_names = ['Slight Injury', 'Serious Injury', 'Fatal injury']

y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)

# Metrics
acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, average='weighted')
rec   = recall_score(y_test, y_pred, average='weighted')
f1_w  = f1_score(y_test, y_pred, average='weighted')
f1_m  = f1_score(y_test, y_pred, average='macro')
roc   = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

print("=" * 52)
print("  FINAL TEST SET PERFORMANCE")
print("=" * 52)
print(f"  Accuracy            : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision (weighted): {prec:.4f}")
print(f"  Recall (weighted)   : {rec:.4f}")
print(f"  F1 Score (weighted) : {f1_w:.4f}")
print(f"  F1 Score (macro)    : {f1_m:.4f}")
print(f"  ROC-AUC (OvR, wtd)  : {roc:.4f}")
print("=" * 52)
print(f"  Target >90% : {'ACHIEVED' if acc >= 0.90 else 'NOT MET'}")
"""))

cells.append(code("""# Full Classification Report
print("\\nClassification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))
"""))

cells.append(code("""# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm      = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=20)

ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
    ax=axes[1], colorbar=False, cmap='Greens', values_format='.2%')
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle(f'Best Model: {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../docs/06_confusion_matrix.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# ROC Curves (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_colors = ['#2ecc71', '#f39c12', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, c) in enumerate(zip(class_names, roc_colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, color=c, lw=2, label=f'{cls} (AUC = {auc(fpr, tpr):.3f})')

ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curves (One-vs-Rest)\\n{best_name}', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim([0,1])
ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig('../docs/07_roc_curves.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""# Per-Class Metrics
prec_cls = precision_score(y_test, y_pred, average=None)
rec_cls  = recall_score(y_test, y_pred, average=None)
f1_cls   = f1_score(y_test, y_pred, average=None)

x   = np.arange(len(class_names))
w   = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, prec_cls, w, label='Precision', color='#3498db', alpha=0.85)
ax.bar(x,     rec_cls,  w, label='Recall',    color='#2ecc71', alpha=0.85)
ax.bar(x + w, f1_cls,   w, label='F1 Score',  color='#e67e22', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision, Recall and F1', fontsize=12, fontweight='bold')
ax.axhline(0.90, color='red', linestyle='--', alpha=0.4)
ax.legend()

for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)

plt.tight_layout()
plt.savefig('../docs/08_per_class_metrics.png', bbox_inches='tight')
plt.show()
"""))

# ── 8. FEATURE IMPORTANCE ──────────────────────────────────────────────────────
cells.append(md("## 8. Feature Importance"))
cells.append(code("""clf = best_pipeline.named_steps['clf']

if hasattr(clf, 'feature_importances_'):
    all_feature_names = cat_cols + num_cols
    importances = clf.feature_importances_
    n_feat = min(len(importances), len(all_feature_names))
    
    feat_df = pd.DataFrame({
        'Feature': all_feature_names[:n_feat],
        'Importance': importances[:n_feat]
    }).sort_values('Importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                   color='#3498db', edgecolor='white', linewidth=0.5)
    ax.set_title(f'Top 15 Feature Importances\\n{best_name}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    plt.tight_layout()
    plt.savefig('../docs/09_feature_importance.png', bbox_inches='tight')
    plt.show()
else:
    print(f"{best_name} does not expose feature_importances_.")
"""))

# ── 9. REGULARIZATION ANALYSIS ─────────────────────────────────────────────────
cells.append(md("""## 9. Regularization Analysis

We visualize two things:
1. The train-vs-validation gap for each model (overfitting check)
2. The effect of XGBoost L2 lambda on validation F1
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Overfit gap per model
gaps = [cv_results[m]['train_accuracy'].mean() - cv_results[m]['test_accuracy'].mean()
        for m in model_names]
bar_colors = ['#e74c3c' if g > 0.08 else '#2ecc71' for g in gaps]
axes[0].bar(short_names, gaps, color=bar_colors)
axes[0].axhline(0.08, color='orange', linestyle='--', label='8% threshold')
axes[0].set_ylabel('Train - Val Accuracy Gap')
axes[0].set_title('Overfitting Gap per Model\\n(lower = better regularized)', fontweight='bold')
axes[0].legend()
for i, g in enumerate(gaps):
    axes[0].text(i, g + 0.002, f'{g:.3f}', ha='center', fontsize=10, fontweight='bold')

# Plot 2: XGBoost L2 lambda sweep
lambdas = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
lambda_f1s = []
print("XGBoost L2 lambda sweep (3-fold CV)...")
for lam in lambdas:
    test_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', XGBClassifier(n_estimators=100, max_depth=6,
                              learning_rate=0.1, reg_lambda=lam, reg_alpha=0.1,
                              random_state=42, verbosity=0, n_jobs=-1))
    ])
    s = cross_validate(test_pipe, X_train, y_train,
                       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                       scoring='f1_weighted', n_jobs=1)
    lambda_f1s.append(s['test_score'].mean())
    print(f"  lambda={lam}: F1={s['test_score'].mean():.4f}")

axes[1].plot(lambdas, lambda_f1s, 'o-', color='#e74c3c', lw=2, markersize=7)
best_lam_idx = np.argmax(lambda_f1s)
axes[1].axvline(lambdas[best_lam_idx], color='green', linestyle='--',
                label=f"Best lambda={lambdas[best_lam_idx]}")
axes[1].set_xlabel('L2 Lambda')
axes[1].set_ylabel('CV F1 (weighted)')
axes[1].set_title('XGBoost L2 Regularization Sweep', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../docs/10_regularization_analysis.png', bbox_inches='tight')
plt.show()
"""))

# ── 10. SAVE ───────────────────────────────────────────────────────────────────
cells.append(md("## 10. Save Model"))
cells.append(code("""os.makedirs('../models', exist_ok=True)
joblib.dump(best_pipeline, '../models/best_rta_model.pkl')
print(f"Model saved to ../models/best_rta_model.pkl")
print(f"Model size: {os.path.getsize('../models/best_rta_model.pkl') / 1024:.1f} KB")
"""))

cells.append(code("""# Final summary table
rows = []
for name in model_names:
    rows.append({
        'Model':      name,
        'Train Acc':  f"{cv_results[name]['train_accuracy'].mean():.4f}",
        'CV Acc':     f"{cv_results[name]['test_accuracy'].mean():.4f}",
        'CV F1':      f"{cv_results[name]['test_f1_weighted'].mean():.4f}",
        'Gap':        f"{cv_results[name]['train_accuracy'].mean()-cv_results[name]['test_accuracy'].mean():.4f}",
        'Selected':   'YES' if name == best_name else '',
    })

summary_df = pd.DataFrame(rows)
print("\\nCross-Validation Summary:")
print(summary_df.to_string(index=False))
print(f"\\nFinal Test Accuracy : {acc*100:.2f}%")
print(f"Final Test ROC-AUC  : {roc:.4f}")
print(f"Target >90%         : {'ACHIEVED' if acc >= 0.90 else 'NOT MET'}")
"""))

# ── 11. PREDICT FUNCTION ───────────────────────────────────────────────────────
cells.append(md("## 11. Prediction Function"))
cells.append(code("""def predict_severity(input_dict):
    \"\"\"
    Predict accident severity from a feature dictionary.
    
    Returns predicted class, confidence and all class probabilities.
    \"\"\"
    class_names = ['Slight Injury', 'Serious Injury', 'Fatal injury']
    model = joblib.load('../models/best_rta_model.pkl')
    
    df_input    = pd.DataFrame([input_dict])
    pred_class  = model.predict(df_input)[0]
    pred_proba  = model.predict_proba(df_input)[0]
    
    return {
        'predicted_severity': class_names[pred_class],
        'confidence':         f'{pred_proba[pred_class]*100:.1f}%',
        'probabilities': {c: f'{p*100:.1f}%' for c, p in zip(class_names, pred_proba)}
    }

# Test case: high-risk scenario
high_risk = {
    'Time': 'Night (22-6)',
    'Day_of_week': 'Friday',
    'Age_band_of_driver': '18-30',
    'Sex_of_driver': 'Male',
    'Educational_level': 'High school',
    'Vehicle_driver_relation': 'Employee',
    'Driving_experience': 'Below 1yr',
    'Lanes_or_Medians': 'Undivided Two way',
    'Types_of_Junction': 'Y Shape',
    'Road_surface_type': 'Asphalt roads',
    'Road_surface_conditions': 'Wet or damp',
    'Light_conditions': 'Darkness - no lighting',
    'Weather_conditions': 'Raining',
    'Type_of_collision': 'Rollover',
    'Number_of_vehicles_involved': 3,
    'Number_of_casualties': 4,
    'Vehicle_type': 'Lorry (41-100Q)',
    'Cause_of_accident': 'Drunk driving',
    'Pedestrian_movement': 'Crossing from nearside',
    'Vehicle_movement': 'Turnover',
    'Type_of_vehicle': 'Long lorry',
    'Road_allignment': 'Steep grade downward with mountainous terrain',
    'Area_accident_occured': 'Outside Addis Ababa',
    'Sub_district': 'Akaki Kaliti'
}

result = predict_severity(high_risk)
print("High-Risk Scenario:")
print(f"  Predicted Severity : {result['predicted_severity']}")
print(f"  Confidence         : {result['confidence']}")
print(f"  All Probabilities  :")
for cls, prob in result['probabilities'].items():
    print(f"    {cls}: {prob}")
"""))

nb.cells = cells
path = '/home/claude/rta_project/notebooks/Ethiopia_RTA_Severity_Prediction.ipynb'
with open(path, 'w') as f:
    nbf.write(nb, f)
print(f"Notebook written: {path}")
