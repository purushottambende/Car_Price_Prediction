# -*- coding: utf-8 -*-
"""
transform_used_cars.py
======================
Transforms the new used_cars.csv dataset to match the expected
column structure of car_data.csv for training pipeline compatibility.

Run from the project root:
    python transform_used_cars.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
INPUT_FILE = ROOT / "data" / "used_cars.csv"
OUTPUT_FILE = ROOT / "data" / "car_data_new.csv"

print("=" * 60)
print("TRANSFORMING DATASET")
print("=" * 60)

# Load new dataset
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows from used_cars.csv")
print(f"Columns: {df.columns.tolist()}")

# Create new dataframe with expected columns
df_transformed = pd.DataFrame()

# 1. Car_Name: combine brand + model
df_transformed["Car_Name"] = (df["brand"].fillna("Unknown") + " " + df["model"].fillna("")).str.strip()

# 2. Year: extract from model_year
df_transformed["Year"] = pd.to_numeric(df["model_year"], errors="coerce").fillna(2015).astype(int)

# 3. Selling_Price: clean price column (remove $, commas, convert to lakhs)
# "$10,300" -> 10300 -> 0.103 lakhs (divide by 100,000)
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).replace("$", "").replace(",", "")
    try:
        dollars = float(price_str)
        # Convert USD to INR lakhs (approximate: 1 USD = 83 INR, 1 lakh = 100,000)
        # So dollars * 83 / 100000 = lakhs
        lakhs = (dollars * 83) / 100000
        return lakhs
    except:
        return np.nan

df_transformed["Selling_Price"] = df["price"].apply(clean_price)

# 4. Present_Price: estimate as Selling_Price * 1.2 to 1.5 based on age
# Newer cars have higher ratio, older cars lower
age = 2024 - df_transformed["Year"]
ratio = 1.5 - (age * 0.02)  # decreases with age
ratio = ratio.clip(1.1, 1.5)
df_transformed["Present_Price"] = (df_transformed["Selling_Price"] * ratio).round(2)

# 5. Kms_Driven: clean milage column ("51,000 mi." -> 51000)
def clean_mileage(mile_str):
    if pd.isna(mile_str):
        return np.nan
    mile_str = str(mile_str).replace(",", "").replace(" mi.", "").replace("mi", "").strip()
    try:
        miles = float(mile_str)
        # Convert miles to km (1 mile = 1.609 km)
        return int(miles * 1.609)
    except:
        return np.nan

df_transformed["Kms_Driven"] = df["milage"].apply(clean_mileage)

# 6. Fuel_Type: standardize fuel types
fuel_mapping = {
    "Gasoline": "Petrol",
    "Diesel": "Diesel",
    "Hybrid": "Petrol",
    "E85 Flex Fuel": "Petrol",
    "Electric": "Electric",
    "": "Petrol",
}
df_transformed["Fuel_Type"] = df["fuel_type"].fillna("Gasoline").map(
    lambda x: fuel_mapping.get(x, "Petrol")
)

# 7. Seller_Type: default to "Dealer" (most common in original dataset)
df_transformed["Seller_Type"] = "Dealer"

# 8. Transmission: standardize transmission types
def clean_transmission(trans_str):
    if pd.isna(trans_str):
        return "Manual"
    trans_str = str(trans_str).lower()
    if "automatic" in trans_str or "a/t" in trans_str:
        return "Automatic"
    elif "manual" in trans_str or "m/t" in trans_str:
        return "Manual"
    else:
        return "Automatic"  # default

df_transformed["Transmission"] = df["transmission"].apply(clean_transmission)

# 9. Owner: default to 0 (first owner assumption)
# Could use "accident" field to infer: if accident reported, owner=1, else owner=0
df_transformed["Owner"] = df["accident"].apply(
    lambda x: 0 if "None reported" in str(x) else 1
)

# Remove rows with missing critical columns
print(f"\nBefore cleaning: {len(df_transformed)} rows")
df_transformed = df_transformed.dropna(subset=["Selling_Price", "Year", "Kms_Driven"])
print(f"After removing NaN in critical columns: {len(df_transformed)} rows")

# Remove unrealistic outliers
df_transformed = df_transformed[df_transformed["Selling_Price"] > 0]
df_transformed = df_transformed[df_transformed["Year"] >= 1990]
df_transformed = df_transformed[df_transformed["Year"] <= 2024]
print(f"After removing outliers: {len(df_transformed)} rows")

# Save transformed dataset
df_transformed.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved transformed dataset to {OUTPUT_FILE}")
print(f"  Final shape: {df_transformed.shape}")
print(f"\nSample rows:")
print(df_transformed.head())
print(f"\nColumn summary:")
print(df_transformed.describe())
print("\n" + "=" * 60)
print("TRANSFORMATION COMPLETE")
print("=" * 60)
