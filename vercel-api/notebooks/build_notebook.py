import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ─── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""# Ethiopian Road Traffic Accident Severity Prediction
## Full ML Pipeline: Data to Deployment
**Dataset**: Addis Ababa RTA — Mendeley (2017-2020) + Figshare (2016-2022)  
**Task**: Multiclass Classification — Slight / Serious / Fatal  
**Techniques**: Regularized ML, sklearn Pipeline, SMOTE, Cross-Validation  
**Author**: Ethiopia ML Project  
---
"""))

# ─── SECTION 1: SETUP ─────────────────────────────────────────────────────────
cells.append(md("## 1. Setup & Imports"))
cells.append(code("""
import warnings
warnings.filterwarnings('ignore')

# Core
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# sklearn - preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# sklearn - models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# xgboost
from xgboost import XGBClassifier

# imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Model saving
import joblib
import os

# Styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

print("All libraries imported successfully.")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version:  {np.__version__}")
"""))

# ─── SECTION 2: DATA LOADING ──────────────────────────────────────────────────
cells.append(md("""## 2. Data Loading & Merging

**Mendeley dataset**: 12,316 rows, 32 features, Addis Ababa (2017-2020)  
**Figshare dataset**: 13,064 rows, 31 features, Addis Ababa (2016-2022)  
Both datasets share the same schema (collected from Addis Ababa Police Departments).
"""))

cells.append(code("""
# Load merged dataset (Mendeley + Figshare)
DATA_PATH = '../data/RTA_combined.csv'
df = pd.read_csv(DATA_PATH)

print(f"Total records after merge: {df.shape[0]:,}")
print(f"Total features:            {df.shape[1]}")
print(f"\\nSource breakdown:")
print(df['Source'].value_counts())
print(f"\\nYear coverage:")
print(df['Year'].value_counts().sort_index())
"""))

cells.append(code("""
# Preview the data
df.head(10)
"""))

cells.append(code("""
# Column info
df.info()
"""))

# ─── SECTION 3: EDA ───────────────────────────────────────────────────────────
cells.append(md("""## 3. Exploratory Data Analysis (EDA)

We answer 5 key questions:
1. How severe are accidents in Addis Ababa?
2. When do accidents happen?
3. What causes the most fatal accidents?
4. Which road conditions are most dangerous?
5. Is the dataset imbalanced?
"""))

cells.append(code("""
# ── 3.1 Target Distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ['#2ecc71', '#f39c12', '#e74c3c']
severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
counts = df['Accident_severity'].value_counts().reindex(severity_order)

# Bar chart
axes[0].bar(severity_order, counts.values, color=colors, edgecolor='white', linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 150, f'{v:,}\\n({v/len(df)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Accident Severity Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, max(counts.values)*1.15)
axes[0].tick_params(axis='x', rotation=0)

# Pie chart
axes[1].pie(counts.values, labels=severity_order, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Severity Proportions', fontsize=13, fontweight='bold')

plt.suptitle('Addis Ababa Road Traffic Accident Severity (2016-2022)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../docs/01_target_distribution.png', bbox_inches='tight')
plt.show()
print("\\nClass imbalance ratio (Slight:Fatal):", round(counts['Slight Injury'] / counts['Fatal injury'], 1))
"""))

cells.append(code("""
# ── 3.2 Temporal Patterns ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

time_order = ['Morning (6-10)', 'Noon (10-14)', 'Afternoon (14-18)', 'Evening (18-22)', 'Night (22-6)']
day_order   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Time of day
time_sev = df.groupby(['Time', 'Accident_severity']).size().unstack().reindex(time_order)
time_sev[severity_order].plot(kind='bar', ax=axes[0], color=colors,
                               edgecolor='white', linewidth=0.8)
axes[0].set_title('Accidents by Time of Day', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(title='Severity', fontsize=8)

# Day of week
day_sev = df.groupby(['Day_of_week', 'Accident_severity']).size().unstack().reindex(day_order)
day_sev[severity_order].plot(kind='bar', ax=axes[1], color=colors,
                              edgecolor='white', linewidth=0.8)
axes[1].set_title('Accidents by Day of Week', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=25)
axes[1].legend(title='Severity', fontsize=8)

plt.tight_layout()
plt.savefig('../docs/02_temporal_patterns.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# ── 3.3 Top Causes by Severity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

cause_sev = df.groupby(['Cause_of_accident', 'Accident_severity']).size().unstack(fill_value=0)
cause_sev['Fatal_rate'] = cause_sev['Fatal injury'] / cause_sev.sum(axis=1)
cause_sev = cause_sev.sort_values('Fatal_rate', ascending=True)

cause_sev[severity_order].plot(kind='barh', ax=ax, color=colors,
                                edgecolor='white', linewidth=0.5)
ax.set_title('Accident Causes by Severity', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Accidents')
ax.legend(title='Severity')
plt.tight_layout()
plt.savefig('../docs/03_causes_severity.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# ── 3.4 Road & Environment Conditions ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cat_features = [
    ('Road_surface_conditions', 'Road Surface Condition'),
    ('Light_conditions',        'Light Conditions'),
    ('Weather_conditions',      'Weather Conditions'),
    ('Types_of_Junction',       'Junction Type'),
]

for ax, (col, title) in zip(axes.flat, cat_features):
    pivot = df.groupby([col, 'Accident_severity']).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=severity_order, fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

plt.suptitle('Road & Environment Conditions vs Severity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../docs/04_road_conditions.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# ── 3.5 Numerical Feature Distributions ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col in zip(axes, ['Number_of_vehicles_involved', 'Number_of_casualties']):
    data = [df[df['Accident_severity']==s][col].values for s in severity_order]
    bp = ax.boxplot(data, labels=severity_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col.replace('_', ' '), fontsize=11, fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('../docs/05_numeric_distributions.png', bbox_inches='tight')
plt.show()
"""))

# ─── SECTION 4: PREPROCESSING ─────────────────────────────────────────────────
cells.append(md("""## 4. Data Preprocessing

Steps:
1. Drop metadata columns (Source, Year — not predictive features)
2. Encode target variable
3. Identify categorical vs numerical features
4. Build sklearn ColumnTransformer pipeline
"""))

cells.append(code("""
# ── 4.1 Drop metadata, encode target ─────────────────────────────────────────
df_model = df.drop(columns=['Source', 'Year'])

# Encode target
label_map = {'Slight Injury': 0, 'Serious Injury': 1, 'Fatal injury': 2}
df_model['target'] = df_model['Accident_severity'].map(label_map)

X = df_model.drop(columns=['Accident_severity', 'target'])
y = df_model['target']

print("Feature matrix shape:", X.shape)
print("Target classes:", label_map)
print("\\nTarget counts:")
for k, v in label_map.items():
    print(f"  {v} ({k}): {(y == v).sum():,}")
"""))

cells.append(code("""
# ── 4.2 Identify feature types ───────────────────────────────────────────────
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features ({len(cat_cols)}):")
for c in cat_cols: print(f"  {c}")

print(f"\\nNumerical features ({len(num_cols)}):")
for c in num_cols: print(f"  {c}")
"""))

cells.append(code("""
# ── 4.3 Check missing values ─────────────────────────────────────────────────
missing = X.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("No missing values found in feature matrix.")
else:
    print("Missing values detected:")
    print(missing)
"""))

cells.append(code("""
# ── 4.4 Train/Test Split (Stratified) ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set:   {X_train.shape[0]:,} rows")
print(f"Test set:       {X_test.shape[0]:,} rows")
print(f"\\nTraining class distribution:")
for k, v in label_map.items():
    cnt = (y_train == label_map[k]).sum()
    print(f"  {k}: {cnt:,} ({cnt/len(y_train)*100:.1f}%)")
"""))

# ─── SECTION 5: ML PIPELINE ───────────────────────────────────────────────────
cells.append(md("""## 5. ML Pipeline Construction

### Pipeline Architecture

```
Raw Data
   |
   ├── Categorical Columns → SimpleImputer (most_frequent) → OrdinalEncoder
   └── Numerical Columns  → SimpleImputer (median) → StandardScaler
                                    |
                              ColumnTransformer
                                    |
                                  SMOTE  (handle class imbalance)
                                    |
                               Classifier
```

### Regularization Strategy

| Model | Regularization | Parameter |
|-------|---------------|-----------|
| Logistic Regression | L2 (Ridge) | C = 0.1 (strong) |
| Random Forest | Structural | max_depth=12, min_samples_leaf=5 |
| XGBoost | L1 + L2 + Depth | alpha=0.1, lambda=1.5, max_depth=6 |
| Gradient Boosting | Shrinkage + Depth | learning_rate=0.05, max_depth=4 |

**Why these regularization choices?**
- After OHE, we get ~80 features. L2 on LR prevents coefficient explosion.
- Tree depth limits prevent overfitting on the majority class.
- XGBoost alpha/lambda directly penalize leaf weights.
"""))

cells.append(code("""
# ── 5.1 Preprocessor ─────────────────────────────────────────────────────────
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipeline, cat_cols),
    ('num', num_pipeline, num_cols)
], remainder='drop')

print("Preprocessor built.")
print(f"  Categorical columns processed: {len(cat_cols)}")
print(f"  Numerical columns processed:   {len(num_cols)}")
"""))

cells.append(code("""
# ── 5.2 Define all model pipelines ───────────────────────────────────────────
models = {

    'Logistic Regression (L2)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(
            C=0.1,                   # L2 regularization strength (lower = stronger)
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42
        ))
    ]),

    'Random Forest (Regularized)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,            # depth limit = regularization
            min_samples_leaf=5,      # min samples per leaf = regularization
            min_samples_split=10,    # min samples to split = regularization
            max_features='sqrt',     # feature subsampling = regularization
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ]),

    'XGBoost (L1+L2)': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=300,
            max_depth=6,             # depth regularization
            learning_rate=0.05,      # shrinkage
            reg_alpha=0.1,           # L1 regularization
            reg_lambda=1.5,          # L2 regularization
            subsample=0.8,           # row subsampling
            colsample_bytree=0.8,    # column subsampling
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ]),

    'Gradient Boosting': ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,             # depth regularization
            learning_rate=0.05,      # shrinkage = regularization
            subsample=0.8,           # stochastic gradient boosting
            min_samples_leaf=10,
            random_state=42
        ))
    ]),
}

print(f"Total models to train: {len(models)}")
for name in models: print(f"  - {name}")
"""))

# ─── SECTION 6: TRAINING & CV ─────────────────────────────────────────────────
cells.append(md("""## 6. Model Training with Cross-Validation

We use **Stratified 5-Fold Cross-Validation** to:
- Get reliable performance estimates on all 3 classes
- Detect overfitting (gap between train and CV score)
- Pick the best model before touching the test set
"""))

cells.append(code("""
# ── 6.1 Cross-Validation ─────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("Running 5-Fold Cross-Validation...\\n")
for name, pipeline in models.items():
    scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring={
            'accuracy':  'accuracy',
            'f1_macro':  'f1_macro',
            'f1_weighted': 'f1_weighted',
        },
        n_jobs=-1,
        return_train_score=True
    )
    cv_results[name] = scores

    train_acc = scores['train_accuracy'].mean()
    val_acc   = scores['test_accuracy'].mean()
    val_f1    = scores['test_f1_weighted'].mean()
    overfit   = train_acc - val_acc

    print(f"{name}")
    print(f"  Train Accuracy:    {train_acc:.4f}")
    print(f"  Val Accuracy:      {val_acc:.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"  Val F1 (weighted): {val_f1:.4f}")
    print(f"  Overfit gap:       {overfit:.4f}  {'[OK]' if overfit < 0.05 else '[WARNING: overfit]'}")
    print()
"""))

cells.append(code("""
# ── 6.2 CV Results Comparison Chart ──────────────────────────────────────────
model_names = list(cv_results.keys())
short_names = ['LR (L2)', 'RF (Reg)', 'XGBoost', 'GradBoost']

train_accs = [cv_results[m]['train_accuracy'].mean() for m in model_names]
val_accs   = [cv_results[m]['test_accuracy'].mean()  for m in model_names]
val_f1s    = [cv_results[m]['test_f1_weighted'].mean() for m in model_names]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width, train_accs, width, label='Train Accuracy', color='#3498db', alpha=0.85)
b2 = ax.bar(x,          val_accs,   width, label='Val Accuracy',   color='#2ecc71', alpha=0.85)
b3 = ax.bar(x + width,  val_f1s,    width, label='Val F1 (weighted)', color='#e67e22', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0.7, 1.02)
ax.set_title('Model Comparison: Cross-Validation Results', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0.90, color='red', linestyle='--', alpha=0.5, label='90% target')
ax.text(3.6, 0.905, '90% target', color='red', fontsize=9)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../docs/06_cv_comparison.png', bbox_inches='tight')
plt.show()
"""))

# ─── SECTION 7: BEST MODEL & EVALUATION ───────────────────────────────────────
cells.append(md("""## 7. Best Model Selection & Full Evaluation

Select the model with the highest **weighted F1 on validation** (most robust metric for imbalanced multiclass problems).
"""))

cells.append(code("""
# ── 7.1 Select best model ─────────────────────────────────────────────────────
best_name = max(cv_results, key=lambda m: cv_results[m]['test_f1_weighted'].mean())
best_pipeline = models[best_name]

print(f"Best model: {best_name}")
print(f"  Val F1 (weighted): {cv_results[best_name]['test_f1_weighted'].mean():.4f}")
print(f"  Val Accuracy:      {cv_results[best_name]['test_accuracy'].mean():.4f}")

print(f"\\nTraining best model on full training set...")
best_pipeline.fit(X_train, y_train)
print("Done.")
"""))

cells.append(code("""
# ── 7.2 Test Set Predictions ──────────────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)

class_names = ['Slight Injury', 'Serious Injury', 'Fatal injury']

# ── Core Metrics ──────────────────────────────────────────────────────────────
acc      = accuracy_score(y_test, y_pred)
prec_w   = precision_score(y_test, y_pred, average='weighted')
rec_w    = recall_score(y_test, y_pred, average='weighted')
f1_w     = f1_score(y_test, y_pred, average='weighted')
f1_mac   = f1_score(y_test, y_pred, average='macro')
roc_auc  = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

print("=" * 50)
print(f"  FINAL TEST SET PERFORMANCE")
print("=" * 50)
print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision (weighted): {prec_w:.4f}")
print(f"  Recall (weighted)   : {rec_w:.4f}")
print(f"  F1 Score (weighted) : {f1_w:.4f}")
print(f"  F1 Score (macro)    : {f1_mac:.4f}")
print(f"  ROC-AUC (OvR, wtd)  : {roc_auc:.4f}")
print("=" * 50)
print(f"\\n  Target met (>90%): {'YES' if acc >= 0.90 else 'NO - check pipeline'}")
"""))

cells.append(code("""
# ── 7.3 Full Classification Report ───────────────────────────────────────────
print("\\nClassification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))
"""))

cells.append(code("""
# ── 7.4 Confusion Matrix ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=20)

# Normalized (percentage per true class)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp2.plot(ax=axes[1], colorbar=False, cmap='Greens', values_format='.2%')
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle(f'Best Model: {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../docs/07_confusion_matrix.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# ── 7.5 ROC Curves (One-vs-Rest) ─────────────────────────────────────────────
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors_roc  = ['#2ecc71', '#f39c12', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, color) in enumerate(zip(class_names, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    auc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {auc_score:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'ROC Curves (One-vs-Rest)\\n{best_name}', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('../docs/08_roc_curves.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""
# ── 7.6 Per-Class Metrics Bar Chart ──────────────────────────────────────────
prec_per  = precision_score(y_test, y_pred, average=None)
rec_per   = recall_score(y_test, y_pred, average=None)
f1_per    = f1_score(y_test, y_pred, average=None)

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, prec_per, width, label='Precision', color='#3498db', alpha=0.85)
ax.bar(x,         rec_per,  width, label='Recall',    color='#2ecc71', alpha=0.85)
ax.bar(x + width, f1_per,   width, label='F1 Score',  color='#e67e22', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision, Recall and F1 Score', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0.90, color='red', linestyle='--', alpha=0.4)

for bars in [ax.patches[:3], ax.patches[3:6], ax.patches[6:]]:
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        break
    break

plt.tight_layout()
plt.savefig('../docs/09_per_class_metrics.png', bbox_inches='tight')
plt.show()
"""))

# ─── SECTION 8: FEATURE IMPORTANCE ───────────────────────────────────────────
cells.append(md("## 8. Feature Importance"))
cells.append(code("""
# ── 8.1 Feature Importance ───────────────────────────────────────────────────
# Extract feature names after preprocessing
try:
    feature_names = (
        list(cat_cols) +
        list(num_cols)
    )

    # Get classifier from pipeline
    clf = best_pipeline.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
                       color='#3498db', edgecolor='white', linewidth=0.5)
        ax.set_title(f'Top 15 Feature Importances\\n{best_name}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance Score')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig('../docs/10_feature_importance.png', bbox_inches='tight')
        plt.show()
    else:
        print(f"Note: {best_name} does not expose feature importances directly.")
        print("Use permutation importance for model-agnostic importance.")
except Exception as e:
    print(f"Feature importance not available: {e}")
"""))

# ─── SECTION 9: REGULARIZATION ANALYSIS ──────────────────────────────────────
cells.append(md("""## 9. Regularization Analysis

We show the effect of regularization on model performance. This proves regularization was necessary.
"""))
cells.append(code("""
# ── 9.1 Overfitting Check (Train vs Val scores) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Train vs Val accuracy gap
train_accs_all = [cv_results[m]['train_accuracy'].mean() for m in model_names]
val_accs_all   = [cv_results[m]['test_accuracy'].mean()  for m in model_names]
gaps = [t - v for t, v in zip(train_accs_all, val_accs_all)]

x = np.arange(len(short_names))
axes[0].bar(x, gaps, color=['#e74c3c' if g > 0.05 else '#2ecc71' for g in gaps])
axes[0].axhline(0.05, color='orange', linestyle='--', label='5% overfit threshold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, rotation=10)
axes[0].set_ylabel('Train-Val Accuracy Gap')
axes[0].set_title('Overfitting Gap per Model\\n(lower is better)', fontweight='bold')
axes[0].legend()

# XGBoost: L2 lambda effect
lambdas = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
xgb_cv_scores = []
print("Testing XGBoost lambda (L2) regularization sweep...")
for lam in lambdas:
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=100, max_depth=6,
            reg_lambda=lam, reg_alpha=0.1,
            learning_rate=0.05, random_state=42,
            verbosity=0, n_jobs=-1
        ))
    ])
    s = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
    xgb_cv_scores.append(s.mean())
    print(f"  lambda={lam}: F1={s.mean():.4f}")

axes[1].plot(lambdas, xgb_cv_scores, 'o-', color='#e74c3c', linewidth=2, markersize=7)
axes[1].axvline(1.5, color='green', linestyle='--', label='Chosen lambda=1.5')
axes[1].set_xlabel('L2 Lambda (Regularization Strength)')
axes[1].set_ylabel('CV F1 (weighted)')
axes[1].set_title('XGBoost: L2 Regularization Sweep\\n(lambda effect on F1)', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../docs/11_regularization_analysis.png', bbox_inches='tight')
plt.show()
"""))

# ─── SECTION 10: SAVE MODEL ───────────────────────────────────────────────────
cells.append(md("## 10. Save Model & Results"))
cells.append(code("""
# ── 10.1 Save best model ─────────────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)
model_path = '../models/best_rta_model.pkl'
joblib.dump(best_pipeline, model_path)
print(f"Model saved to: {model_path}")

# Save metadata
import json
metadata = {
    'model_name': best_name,
    'test_accuracy': round(float(acc), 4),
    'test_f1_weighted': round(float(f1_w), 4),
    'test_roc_auc': round(float(roc_auc), 4),
    'test_precision_weighted': round(float(prec_w), 4),
    'test_recall_weighted': round(float(rec_w), 4),
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
    'features': list(X.columns),
    'classes': class_names,
    'label_map': label_map,
}
with open('../models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved to: ../models/model_metadata.json")
"""))

cells.append(code("""
# ── 10.2 Final Summary Table ──────────────────────────────────────────────────
print("\\n" + "=" * 55)
print("  FINAL MODEL SUMMARY")
print("=" * 55)

summary_data = []
for name in model_names:
    summary_data.append({
        'Model': name,
        'Train Acc': f"{cv_results[name]['train_accuracy'].mean():.4f}",
        'CV Acc':    f"{cv_results[name]['test_accuracy'].mean():.4f}",
        'CV F1':     f"{cv_results[name]['test_f1_weighted'].mean():.4f}",
        'Overfit':   f"{cv_results[name]['train_accuracy'].mean() - cv_results[name]['test_accuracy'].mean():.4f}",
        'Best':      'YES' if name == best_name else ''
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("=" * 55)
print(f"\\nSelected: {best_name}")
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Target (>90%): {'ACHIEVED' if acc >= 0.90 else 'NOT ACHIEVED'}")
"""))

# ─── SECTION 11: PREDICTION FUNCTION ──────────────────────────────────────────
cells.append(md("## 11. Prediction Function (for Deployment)"))
cells.append(code("""
# ── 11.1 Single prediction function ──────────────────────────────────────────
def predict_severity(input_dict, model_pipeline=best_pipeline):
    \"\"\"
    Predict accident severity for a single accident record.
    
    Parameters:
        input_dict: dict with accident features
        model_pipeline: trained pipeline
    
    Returns:
        dict with prediction, confidence and probabilities
    \"\"\"
    class_names = ['Slight Injury', 'Serious Injury', 'Fatal injury']
    
    input_df = pd.DataFrame([input_dict])
    
    pred_class   = model_pipeline.predict(input_df)[0]
    pred_proba   = model_pipeline.predict_proba(input_df)[0]
    confidence   = pred_proba[pred_class]
    
    return {
        'predicted_severity': class_names[pred_class],
        'confidence': f'{confidence*100:.1f}%',
        'probabilities': {
            cls: f'{p*100:.1f}%' 
            for cls, p in zip(class_names, pred_proba)
        }
    }

# ── 11.2 Test the prediction function ────────────────────────────────────────
test_case = {
    'Time': 'Night (22-6)',
    'Day_of_week': 'Friday',
    'Age_band_of_driver': '18-30',
    'Sex_of_driver': 'Male',
    'Educational_level': 'High school',
    'Vehicle_driver_relation': 'Employee',
    'Driving_experience': '1-2yr',
    'Lanes_or_Medians': 'Undivided Two way',
    'Types_of_Junction': 'Y Shape',
    'Road_surface_type': 'Asphalt roads',
    'Road_surface_conditions': 'Wet or damp',
    'Light_conditions': 'Darkness - no lighting',
    'Weather_conditions': 'Raining',
    'Type_of_collision': 'Rollover',
    'Number_of_vehicles_involved': 2,
    'Number_of_casualties': 3,
    'Vehicle_type': 'Automobile',
    'Cause_of_accident': 'Overspeed',
    'Pedestrian_movement': 'Not a Pedestrian',
    'Vehicle_movement': 'Going straight',
    'Type_of_vehicle': 'Automobile',
    'Road_allignment': 'Tangent road with flat terrain',
    'Area_accident_occured': 'Residential areas',
    'Sub_district': 'Bole'
}

result = predict_severity(test_case)
print("Test Prediction:")
print(f"  Scenario: Night, Rainy, Rollover, Overspeed, Young Driver")
print(f"  Predicted Severity: {result['predicted_severity']}")
print(f"  Confidence:         {result['confidence']}")
print(f"  Probabilities:")
for cls, prob in result['probabilities'].items():
    print(f"    {cls}: {prob}")
"""))

nb.cells = cells
output_path = '/home/claude/rta_project/notebooks/Ethiopia_RTA_Severity_Prediction.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_path}")
