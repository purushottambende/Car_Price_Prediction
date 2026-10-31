"""
src/preprocessing.py
====================
Custom scikit-learn transformers and helper utilities for the
Second-Hand Car Price Prediction project.

All classes and functions are at module level (no lambdas) so that
joblib can pickle / unpickle pipelines that use them correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Column name constants (match the raw CSV exactly)
# ---------------------------------------------------------------------------
COL_CAR_NAME    = "Car_Name"
COL_YEAR        = "Year"
COL_SELLING     = "Selling_Price"
COL_PRESENT     = "Present_Price"
COL_KMS         = "Kms_Driven"
COL_FUEL        = "Fuel_Type"
COL_SELLER      = "Seller_Type"
COL_TRANS       = "Transmission"
COL_OWNER       = "Owner"

# Derived columns
COL_BRAND       = "Brand"
COL_CAR_AGE     = "Car_Age"
COL_KM_PER_YR   = "Km_Per_Year"

CURRENT_YEAR    = 2024   # fixed reference year so predictions are reproducible

# ---------------------------------------------------------------------------
# 1. BrandExtractor
#    Extracts the first word of Car_Name as the Brand feature.
# ---------------------------------------------------------------------------
class BrandExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts `Brand` column from the first token of `Car_Name`.
    Drops `Car_Name` afterwards.
    Works on a DataFrame copy so the original is never mutated.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if COL_CAR_NAME in X.columns:
            X[COL_BRAND] = (
                X[COL_CAR_NAME]
                .astype(str)
                .str.strip()
                .str.split()
                .str[0]
                .str.lower()
                .str.capitalize()
            )
            X = X.drop(columns=[COL_CAR_NAME])
        return X


# ---------------------------------------------------------------------------
# 2. CarAgeFeatureEngineer
#    Adds Car_Age = CURRENT_YEAR - Year
#    Adds Km_Per_Year = Kms_Driven / max(Car_Age, 1)
# ---------------------------------------------------------------------------
class CarAgeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds `Car_Age` and `Km_Per_Year` derived features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        year = pd.to_numeric(X[COL_YEAR], errors="coerce").fillna(CURRENT_YEAR)
        X[COL_CAR_AGE] = (CURRENT_YEAR - year).clip(lower=0)
        kms  = pd.to_numeric(X[COL_KMS], errors="coerce").fillna(0)
        X[COL_KM_PER_YR] = kms / X[COL_CAR_AGE].replace(0, 1)
        return X


# ---------------------------------------------------------------------------
# 3. NumericCleaner
#    Coerces all specified columns to numeric; NaN left for imputation.
# ---------------------------------------------------------------------------
class NumericCleaner(BaseEstimator, TransformerMixin):
    """
    Converts a list of columns to numeric dtype.
    Strings like '1248 CC' or '19.7 kmpl' are stripped of non-numeric
    characters before conversion.
    """

    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.extract(r"([\d.]+)", expand=False)
                    .astype(float)
                )
        return X


# ---------------------------------------------------------------------------
# 4. ColumnSelector
#    Selects a fixed list of columns (keeps pipeline deterministic).
# ---------------------------------------------------------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects only the specified columns from a DataFrame."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        present = [c for c in self.columns if c in X.columns]
        return X[present]


# ---------------------------------------------------------------------------
# 5. DataFrameImputer
#    Median for numeric columns, mode for categorical columns.
# ---------------------------------------------------------------------------
class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values:
      - numeric columns  → median (computed at fit time)
      - object/category  → mode  (computed at fit time)
    """

    def fit(self, X, y=None):
        self.fill_values_ = {}
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "category":
                mode_vals = X[col].mode()
                self.fill_values_[col] = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
            else:
                self.fill_values_[col] = float(X[col].median())
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# ---------------------------------------------------------------------------
# 6. Standalone helper – build_feature_list
#    Returns the ordered list of raw features fed into the pipeline.
# ---------------------------------------------------------------------------
def build_feature_list():
    """
    Returns the list of raw feature columns that the user provides.
    Selling_Price is intentionally excluded (target / leakage risk).
    """
    return [
        COL_CAR_NAME,      # will be converted to Brand inside pipeline
        COL_YEAR,
        COL_PRESENT,
        COL_KMS,
        COL_FUEL,
        COL_SELLER,
        COL_TRANS,
        COL_OWNER,
    ]


def build_numeric_features():
    """Numeric feature columns after feature engineering."""
    return [
        COL_YEAR,
        COL_PRESENT,
        COL_KMS,
        COL_OWNER,
        COL_CAR_AGE,
        COL_KM_PER_YR,
    ]


def build_categorical_features():
    """Categorical feature columns used for OneHotEncoding."""
    return [
        COL_BRAND,
        COL_FUEL,
        COL_SELLER,
        COL_TRANS,
    ]


# ---------------------------------------------------------------------------
# 7. IQR outlier bounds helper (used in notebook, not in pipeline)
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = 1.5):
    """Returns (lower_bound, upper_bound) using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ---------------------------------------------------------------------------
# 8. format_inr  – format price in Indian rupee notation (used by app)
# ---------------------------------------------------------------------------
def format_inr(amount_lakhs: float) -> str:
    """
    Converts a price in Lakhs (float) to a nicely formatted ₹ string.
    E.g. 5.75 → '₹ 5,75,000'
    """
    rupees = int(round(amount_lakhs * 1e5))
    # Indian comma formatting: last 3 digits, then groups of 2
    s = str(abs(rupees))
    if len(s) <= 3:
        formatted = s
    else:
        formatted = s[-3:]
        s = s[:-3]
        while s:
            formatted = s[-2:] + "," + formatted
            s = s[:-2]
    sign = "-" if rupees < 0 else ""
    return f"₹ {sign}{formatted}"


# ---------------------------------------------------------------------------
# 9. deal_rating – classify deal quality
# ---------------------------------------------------------------------------
def deal_rating(predicted_lakhs: float, listed_lakhs: float) -> str:
    """
    Returns 'Good Deal', 'Fair Deal', or 'Overpriced' based on how
    the listed price compares to the predicted price.
    """
    if listed_lakhs <= 0 or predicted_lakhs <= 0:
        return "N/A"
    ratio = (listed_lakhs - predicted_lakhs) / predicted_lakhs
    if ratio < -0.10:
        return "Good Deal 🟢"
    elif ratio <= 0.10:
        return "Fair Deal 🟡"
    else:
        return "Overpriced 🔴"
