# -*- coding: utf-8 -*-
"""
train_models.py
===============
Comprehensive ML pipeline for used car price prediction.

This script performs:
  1. Data loading and validation
  2. Exploratory Data Analysis (EDA) with visualization
  3. Feature engineering and preprocessing
  4. Model training (Regression + Classification)
  5. Model evaluation and persistence
  6. Metadata generation for deployment

Models:
  - Regression: Polynomial Regression (degree 2) for price prediction
  - Classification: Decision Tree for price category classification

Usage:
    python train_models.py

Output:
  - models/regressor.joblib
  - models/classifier.joblib  
  - models/metadata.json
  - assets/*.png (visualization graphs)
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure src/ module is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

from src.preprocessing import (
    BrandExtractor, 
    CarAgeFeatureEngineer,
    iqr_bounds,
    COL_CAR_NAME, COL_YEAR, COL_SELLING, COL_PRESENT,
    COL_KMS, COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER,
    COL_BRAND, COL_CAR_AGE, COL_KM_PER_YR, CURRENT_YEAR
)

# Configuration constants
RANDOM_STATE = 42  # For reproducible results
TEST_SIZE = 0.2    # 80-20 train-test split
CV_FOLDS = 5       # Cross-validation folds

# Directory setup
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

DATA_PATH = ROOT / "data" / "car_data_new.csv"
df_raw = pd.read_csv(DATA_PATH)
print(f"Shape: {df_raw.shape}")
print("\nColumn dtypes:\n", df_raw.dtypes)
print("\nMissing values:\n", df_raw.isnull().sum())
print("\nSample:\n", df_raw.head())

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASIC DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASIC DATA CLEANING")
print("=" * 60)

df = df_raw.copy()

# Convert numeric columns to proper data types
numeric_cols = [COL_YEAR, COL_SELLING, COL_PRESENT, COL_KMS, COL_OWNER]
for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )

# Impute missing values - median for numeric, mode for categorical
for col in [COL_YEAR, COL_PRESENT, COL_KMS, COL_OWNER]:
    df[col] = df[col].fillna(df[col].median())
    
for col in [COL_FUEL, COL_SELLER, COL_TRANS]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows where target variable (Selling_Price) is missing
df = df.dropna(subset=[COL_SELLING])
print(f"Shape after cleaning: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Extract brand name from car name (first word)
df[COL_BRAND] = df[COL_CAR_NAME].str.strip().str.split().str[0].str.lower().str.capitalize()

# Calculate car age from manufacturing year
df[COL_CAR_AGE] = (CURRENT_YEAR - df[COL_YEAR]).clip(lower=0)

# Calculate average kilometers driven per year
df[COL_KM_PER_YR] = df[COL_KMS] / df[COL_CAR_AGE].replace(0, 1)

print(f"Added features: {COL_BRAND}, {COL_CAR_AGE}, {COL_KM_PER_YR}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS (EDA) - VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERATING EDA VISUALIZATIONS")
print("=" * 60)

plt.style.use("seaborn-v0_8-whitegrid")

# Graph 1: Price vs Year and Kilometers Driven
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df[COL_YEAR], df[COL_SELLING], alpha=0.5, color="#4C72B0")
axes[0].set_title("Price vs Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Selling Price (Lakhs)")

axes[1].scatter(df[COL_KMS], df[COL_SELLING], alpha=0.5, color="#DD8452")
axes[1].set_title("Price vs Kms Driven")
axes[1].set_xlabel("Kms Driven")
axes[1].set_ylabel("Selling Price (Lakhs)")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "price_vs_year_kms.png", dpi=120)
plt.close()
print("  ✓ Saved price_vs_year_kms.png")

# Graph 2: Average Price by Brand
brand_price = df.groupby(COL_BRAND)[COL_SELLING].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 5))
brand_price.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Average Selling Price by Brand")
ax.set_ylabel("Avg Price (Lakhs)")
ax.set_xlabel("Brand")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "avg_price_by_brand.png", dpi=120)
plt.close()
print("  ✓ Saved avg_price_by_brand.png")

# Graph 3: Average Price by Fuel Type & Transmission
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.groupby(COL_FUEL)[COL_SELLING].mean().sort_values().plot(
    kind="barh", ax=axes[0], color="#55A868")
axes[0].set_title("Avg Price by Fuel Type")

df.groupby(COL_TRANS)[COL_SELLING].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="#C44E52")
axes[1].set_title("Avg Price by Transmission")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "avg_price_fuel_trans.png", dpi=120)
plt.close()
print("  ✓ Saved avg_price_fuel_trans.png")

# Graph 4: Correlation Heatmap
num_cols = [COL_YEAR, COL_SELLING, COL_PRESENT, COL_KMS, COL_OWNER, COL_CAR_AGE, COL_KM_PER_YR]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "correlation_heatmap.png", dpi=120)
plt.close()
print("  ✓ Saved correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. OUTLIER DETECTION AND REMOVAL (IQR Method)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OUTLIER HANDLING")
print("=" * 60)

rows_before = len(df)

# Apply IQR-based outlier filtering to key numeric columns
for col in [COL_SELLING, COL_KMS, COL_PRESENT]:
    lower_bound, upper_bound = iqr_bounds(df[col])
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

rows_after = len(df)
rows_removed = rows_before - rows_after

print(f"  Rows before: {rows_before}")
print(f"  Rows after:  {rows_after}")
print(f"  Removed:     {rows_removed} ({rows_removed/rows_before*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CREATE TARGET VARIABLE FOR CLASSIFICATION (Price Category)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CREATING PRICE CATEGORIES")
print("=" * 60)

# Create tertile-based categories: Low, Medium, High
df["price_category"] = pd.qcut(
    df[COL_SELLING], 
    q=3, 
    labels=["Low", "Medium", "High"]
)

print("Price category distribution:")
print(df["price_category"].value_counts().sort_index())

# Store category boundaries for metadata
q33 = float(df[COL_SELLING].quantile(1/3))
q67 = float(df[COL_SELLING].quantile(2/3))
print(f"\nCategory thresholds:")
print(f"  Low-Medium boundary:   ₹{q33:.2f} Lakhs")
print(f"  Medium-High boundary:  ₹{q67:.2f} Lakhs")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PREPARE FEATURES AND TARGET VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPARING TRAIN-TEST SPLIT")
print("=" * 60)

# Define feature columns (excluding target variable)
FEATURE_COLS = [
    COL_CAR_NAME, COL_YEAR, COL_PRESENT, COL_KMS,
    COL_FUEL, COL_SELLER, COL_TRANS, COL_OWNER
]

X = df[FEATURE_COLS].copy()
y_regression = df[COL_SELLING].values      # Continuous target for regression
y_classification = df["price_category"].values  # Categorical target for classification

# Perform stratified train-test split
X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_regression, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, yc_train, yc_test = train_test_split(
    X, y_classification, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Training samples:   {len(X_train):,}")
print(f"Testing samples:    {len(X_test):,}")
print(f"Train/Test ratio:   {len(X_train)/len(X_test):.1f}:1")

# ══════════════════════════════════════════════════════════════════════════════
# 8. BUILD PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 60)

# Define feature groups
NUM_FEATURES = [COL_YEAR, COL_PRESENT, COL_KMS, COL_OWNER, COL_CAR_AGE, COL_KM_PER_YR]
CAT_FEATURES = [COL_BRAND, COL_FUEL, COL_SELLER, COL_TRANS]

# Numeric preprocessing: Imputation → Scaling
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical preprocessing: Imputation → One-Hot Encoding
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ],
    remainder="drop"
)

print(f"Numeric features ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"Categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}")


def make_base_pipeline(model):
    """
    Create a complete pipeline with preprocessing and model.
    
    Args:
        model: Scikit-learn estimator
        
    Returns:
        Pipeline with brand extraction, feature engineering, preprocessing, and model
    """
    return Pipeline([
        ("brand",      BrandExtractor()),
        ("age",        CarAgeFeatureEngineer()),
        ("preprocess", preprocessor),
        ("model",      model),
    ])

# ══════════════════════════════════════════════════════════════════════════════
# 9. REGRESSION MODEL - Polynomial Regression (Degree 2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING REGRESSION MODEL")
print("=" * 60)
print("Model: Polynomial Regression (degree=2)")

# Build pipeline with polynomial features
pipe_poly = Pipeline([
    ("brand",      BrandExtractor()),
    ("age",        CarAgeFeatureEngineer()),
    ("preprocess", preprocessor),
    ("poly",       PolynomialFeatures(degree=2, include_bias=False)),
    ("model",      LinearRegression()),
])

# Train the model
print("Training...")
pipe_poly.fit(X_train, yr_train)

# Make predictions
y_pred_reg = pipe_poly.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(yr_test, y_pred_reg)
mse = mean_squared_error(yr_test, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, y_pred_reg)
cv_r2 = cross_val_score(pipe_poly, X_train, yr_train, cv=CV_FOLDS, scoring="r2").mean()

# Display results
print(f"\nPerformance Metrics:")
print(f"  MAE (Mean Absolute Error):    ₹{mae:.4f} Lakhs")
print(f"  MSE (Mean Squared Error):     {mse:.4f}")
print(f"  RMSE (Root Mean Squared):     ₹{rmse:.4f} Lakhs")
print(f"  R² Score (Test Set):          {r2:.4f}")
print(f"  R² Score ({CV_FOLDS}-Fold CV):         {cv_r2:.4f}")

# Store results for metadata
reg_result = {
    "model": "Polynomial Regression (deg=2)",
    "MAE": round(mae, 4),
    "MSE": round(mse, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4),
    "CV_R2": round(cv_r2, 4),
}

# Generate R² Score visualization
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(["Polynomial Regression (deg=2)"], [r2], color="#4C72B0", edgecolor="white")
ax.set_xlabel("R² Score")
ax.set_title("Regression Model Performance")
ax.text(r2 + 0.0005, 0, f"{r2:.4f}", va="center", fontsize=10)
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "regression_comparison.png", dpi=120)
plt.close()
print("  ✓ Saved regression_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CLASSIFICATION MODEL - Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING CLASSIFICATION MODEL")
print("=" * 60)
print("Model: Decision Tree Classifier")

LABELS = ["Low", "Medium", "High"]

# Build pipeline with Decision Tree
pipe_dt = make_base_pipeline(
    DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
)

# Train the model
print("Training...")
pipe_dt.fit(X_train, yc_train)

# Make predictions
y_pred_clf = pipe_dt.predict(X_test)

# Calculate evaluation metrics
acc = accuracy_score(yc_test, y_pred_clf)
rpt = classification_report(yc_test, y_pred_clf, output_dict=True, zero_division=0)
macro_p = rpt["macro avg"]["precision"]
macro_r = rpt["macro avg"]["recall"]
macro_f1 = rpt["macro avg"]["f1-score"]

# Calculate ROC-AUC score
try:
    y_prob = pipe_dt.predict_proba(X_test)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(yc_test, classes=pipe_dt.classes_)
    auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
except Exception:
    auc = float("nan")

# Display results
print(f"\nPerformance Metrics:")
print(f"  Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision (Macro):   {macro_p:.4f}")
print(f"  Recall (Macro):      {macro_r:.4f}")
print(f"  F1 Score (Macro):    {macro_f1:.4f}")
print(f"  ROC-AUC (Macro):     {auc:.4f}")

# Store results for metadata
clf_result = {
    "model": "Decision Tree",
    "Accuracy": round(acc, 4),
    "Macro_Precision": round(macro_p, 4),
    "Macro_Recall": round(macro_r, 4),
    "Macro_F1": round(macro_f1, 4),
    "ROC_AUC": round(auc, 4) if not np.isnan(auc) else None,
}

# Generate Confusion Matrix visualization
cm = confusion_matrix(yc_test, y_pred_clf, labels=LABELS)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
ax.set_title("Confusion Matrix — Decision Tree")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print("  ✓ Saved confusion_matrix.png")

# Generate ROC Curves visualization
fig, ax = plt.subplots(figsize=(8, 6))
color_map = {"Low": "#4C72B0", "Medium": "#DD8452", "High": "#55A868"}
clf_classes = list(pipe_dt.classes_)

for label in LABELS:
    try:
        if label in clf_classes:
            col_idx = clf_classes.index(label)
            y_prob_col = pipe_dt.predict_proba(X_test)[:, col_idx]
            y_true_binary = (np.array(yc_test) == label).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_col)
            auc_val = roc_auc_score(y_true_binary, y_prob_col)
            ax.plot(fpr, tpr, color=color_map.get(label, "#333333"),
                    label=f"{label} (AUC={auc_val:.2f})")
    except Exception:
        pass

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Decision Tree")
ax.legend()
plt.tight_layout()
plt.savefig(ASSETS_DIR / "roc_curves.png", dpi=120)
plt.close()
print("  ✓ Saved roc_curves.png")

# Generate Accuracy visualization
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(["Decision Tree"], [acc], color="#55A868", edgecolor="white")
ax.set_xlabel("Accuracy")
ax.set_title("Classification Model Performance")
ax.text(acc + 0.002, 0, f"{acc:.4f}", va="center", fontsize=10)
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "classification_comparison.png", dpi=120)
plt.close()
print("  ✓ Saved classification_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE TRAINED MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAVING TRAINED MODELS")
print("=" * 60)

# Save models with compression
joblib.dump(pipe_poly, MODELS_DIR / "regressor.joblib", compress=3)
joblib.dump(pipe_dt, MODELS_DIR / "classifier.joblib", compress=3)

# Display saved file sizes
for fname in ["regressor.joblib", "classifier.joblib"]:
    file_size_kb = (MODELS_DIR / fname).stat().st_size / 1024
    print(f"  ✓ {fname:<25} {file_size_kb:>6.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# 12. GENERATE DEPLOYMENT METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERATING METADATA")
print("=" * 60)

# Extract unique values and ranges for web app dropdowns/validation
brands_list = sorted(df[COL_BRAND].dropna().unique().tolist())
fuels_list = sorted(df[COL_FUEL].dropna().unique().tolist())
sellers_list = sorted(df[COL_SELLER].dropna().unique().tolist())
trans_list = sorted(df[COL_TRANS].dropna().unique().tolist())
owners_list = sorted(df[COL_OWNER].dropna().unique().tolist())

metadata = {
    "sklearn_version": sklearn.__version__,
    "current_year": CURRENT_YEAR,
    "feature_list": FEATURE_COLS,
    "categorical_options": {
        COL_BRAND:  brands_list,
        COL_FUEL:   fuels_list,
        COL_SELLER: sellers_list,
        COL_TRANS:  trans_list,
        COL_OWNER:  [int(o) for o in owners_list],
    },
    "numeric_ranges": {
        COL_YEAR:    {
            "min": int(df[COL_YEAR].min()),
            "max": int(df[COL_YEAR].max()),
            "default": int(df[COL_YEAR].median())
        },
        COL_PRESENT: {
            "min": float(df[COL_PRESENT].min()),
            "max": float(df[COL_PRESENT].max()),
            "default": float(df[COL_PRESENT].median())
        },
        COL_KMS: {
            "min": float(df[COL_KMS].min()),
            "max": float(df[COL_KMS].max()),
            "default": float(df[COL_KMS].median())
        },
        COL_OWNER: {
            "min": int(df[COL_OWNER].min()),
            "max": int(df[COL_OWNER].max()),
            "default": int(df[COL_OWNER].median())
        },
    },
    "category_thresholds": {
        "Low_to_Medium_lakhs": round(q33, 2),
        "Medium_to_High_lakhs": round(q67, 2),
    },
    "regression_results": [reg_result],
    "classification_results": [clf_result],
    "best_regressor": "Polynomial Regression (deg=2)",
    "best_classifier": "Decision Tree",
}

# Save metadata as JSON
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata.json")
print(f"     - {len(brands_list)} unique brands")
print(f"     - {len(fuels_list)} fuel types")
print(f"     - {len(trans_list)} transmission types")

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE - ALL MODELS AND ASSETS SAVED")
print("=" * 60)
print(f"\nNext steps:")
print(f"  1. Review assets/*.png for visualizations")
print(f"  2. Run 'streamlit run app.py' to launch web application")
print(f"  3. Test models with 'python test_models.py'")
print("=" * 60)
