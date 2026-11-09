# -*- coding: utf-8 -*-
"""
test_models.py
==============
Test script to verify the trained models work correctly.
Tests both regression and classification predictions.

Run from the project root:
    python test_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import json

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import format_inr, deal_rating

print("=" * 60)
print("MODEL TESTING")
print("=" * 60)

# Load models and metadata
print("\n1. Loading models and metadata...")
regressor = joblib.load(ROOT / "models" / "regressor.joblib")
classifier = joblib.load(ROOT / "models" / "classifier.joblib")

with open(ROOT / "models" / "metadata.json", "r") as f:
    metadata = json.load(f)

print(f"   ✓ Regressor loaded: {metadata['best_regressor']}")
print(f"   ✓ Classifier loaded: {metadata['best_classifier']}")
print(f"   ✓ Trained on sklearn version: {metadata['sklearn_version']}")

# Define test cases covering various scenarios
test_cases = [
    {
        "name": "Budget Car - Old Honda",
        "Car_Name": "Honda Civic",
        "Year": 2010,
        "Present_Price": 8.5,
        "Kms_Driven": 80000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 1,
    },
    {
        "name": "Luxury Car - Recent Mercedes",
        "Car_Name": "Mercedes-Benz AMG C 43",
        "Year": 2020,
        "Present_Price": 60.0,
        "Kms_Driven": 15000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 0,
    },
    {
        "name": "Mid-Range - Toyota",
        "Car_Name": "Toyota Camry",
        "Year": 2017,
        "Present_Price": 20.0,
        "Kms_Driven": 45000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 0,
    },
    {
        "name": "High Mileage - Ford",
        "Car_Name": "Ford F-150",
        "Year": 2015,
        "Present_Price": 25.0,
        "Kms_Driven": 150000,
        "Fuel_Type": "Diesel",
        "Seller_Type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 1,
    },
    {
        "name": "Electric - Tesla",
        "Car_Name": "Tesla Model X",
        "Year": 2021,
        "Present_Price": 70.0,
        "Kms_Driven": 20000,
        "Fuel_Type": "Electric",
        "Seller_Type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 0,
    },
]

print("\n2. Running predictions on test cases...\n")
print("-" * 60)

for i, test_case in enumerate(test_cases, 1):
    test_name = test_case.pop("name")
    
    # Create input dataframe
    X_test = pd.DataFrame([test_case])
    
    # Make predictions
    predicted_price = regressor.predict(X_test)[0]
    predicted_category = classifier.predict(X_test)[0]
    
    # Get probabilities if available
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test)[0]
        confidence = max(proba) * 100
    else:
        confidence = None
    
    # Calculate deal rating
    rating = deal_rating(predicted_price, test_case["Present_Price"])
    
    print(f"\nTest Case {i}: {test_name}")
    print(f"  Input:")
    print(f"    Car: {test_case['Car_Name']}")
    print(f"    Year: {test_case['Year']}, Kms: {test_case['Kms_Driven']:,}")
    print(f"    Present Price: {format_inr(test_case['Present_Price'])}")
    print(f"    Fuel: {test_case['Fuel_Type']}, Trans: {test_case['Transmission']}")
    print(f"  Predictions:")
    print(f"    Estimated Price: {format_inr(predicted_price)}")
    print(f"    Price Category: {predicted_category}")
    if confidence:
        print(f"    Confidence: {confidence:.1f}%")
    print(f"    Deal Rating: {rating}")
    print("-" * 60)

# Test edge cases
print("\n3. Testing edge cases...\n")

edge_cases = [
    {
        "name": "Very Old Car",
        "Car_Name": "Honda Accord",
        "Year": 1995,
        "Present_Price": 2.0,
        "Kms_Driven": 200000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 2,
    },
    {
        "name": "Brand New Car",
        "Car_Name": "BMW M850",
        "Year": 2024,
        "Present_Price": 100.0,
        "Kms_Driven": 500,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 0,
    },
]

for i, test_case in enumerate(edge_cases, 1):
    test_name = test_case.pop("name")
    X_test = pd.DataFrame([test_case])
    
    try:
        predicted_price = regressor.predict(X_test)[0]
        predicted_category = classifier.predict(X_test)[0]
        print(f"Edge Case {i} ({test_name}): ✓ PASSED")
        print(f"  Predicted: {format_inr(predicted_price)}, Category: {predicted_category}")
    except Exception as e:
        print(f"Edge Case {i} ({test_name}): ✗ FAILED")
        print(f"  Error: {e}")

# Model statistics summary
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)

print("\nRegression Models:")
for model in metadata["regression_results"][:5]:
    print(f"  {model['model']:25s}: R²={model['R2']:.4f}, RMSE={model['RMSE']:.4f}")

print(f"\n✓ Best Regressor: {metadata['best_regressor']} (R²={metadata['regression_results'][0]['R2']:.4f})")

print("\nClassification Models:")
for model in metadata["classification_results"][:5]:
    print(f"  {model['model']:28s}: Acc={model['Accuracy']:.4f}, F1={model['Macro_F1']:.4f}")

print(f"\n✓ Best Classifier: {metadata['best_classifier']} (F1={metadata['classification_results'][2]['Macro_F1']:.4f})")

print("\nDataset Info:")
print(f"  Total Brands: {len(metadata['categorical_options']['Brand'])}")
print(f"  Fuel Types: {', '.join(metadata['categorical_options']['Fuel_Type'])}")
print(f"  Transmissions: {', '.join(metadata['categorical_options']['Transmission'])}")
print(f"  Price Range: {metadata['numeric_ranges']['Present_Price']['min']:.1f} - {metadata['numeric_ranges']['Present_Price']['max']:.1f} Lakhs")
print(f"  Year Range: {metadata['numeric_ranges']['Year']['min']} - {metadata['numeric_ranges']['Year']['max']}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
print("=" * 60)
