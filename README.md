# Second-Hand Car Price Prediction & Classification

A machine-learning course project that predicts the **selling price** of used cars (regression) and classifies the price into **Low / Medium / High** categories (classification), deployed as an interactive Streamlit web app.

---

## How It Works

1. User fills in car details (brand, year, kms driven, fuels type, etc.) in the sidebar.
2. The input passes through a preprocessing pipeline that extracts the brand, computes car age, and encodes categorical features.
3. A trained **Polynomial Regression (degree 2)** model predicts the estimated selling price.
4. A trained **Decision Tree** classifier assigns a price category (Low / Medium / High).
5. If the user provides an asking price, a deal rating (Good Deal / Fair Deal / Overpriced) is shown based on how far it is from the predicted price.

---

## Project Structure

```
car_price_project/
├── data/
│   ├── used_cars.csv             # Raw dataset (4009 rows, 12 columns)
│   └── car_data_new.csv          # Transformed dataset ready for training
├── notebooks/
│   └── car_price.ipynb           # Full EDA + training notebook
├── src/
│   ├── __init__.py
│   └── preprocessing.py          # Custom sklearn transformers & helpers
├── models/
│   ├── regressor.joblib          # Polynomial Regression pipeline
│   ├── classifier.joblib         # Decision Tree pipeline
│   └── metadata.json             # Dropdown options, ranges, metrics
├── assets/                       # Saved EDA and evaluation charts
├── train_models.py               # Standalone training script
├── transform_used_cars.py        # Script to convert used_cars.csv to training format
├── app.py                        # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The project uses the `used_cars.csv` dataset which is transformed into `car_data_new.csv` for training.

| Column | Type | Description |
|---|---|---|
| Car_Name | string | Car brand + model name |
| Year | int | Year of manufacture (1992–2024) |
| Selling_Price | float | Selling price in Lakhs ₹ (target) |
| Present_Price | float | Current showroom price in Lakhs ₹ |
| Kms_Driven | int | Odometer reading in km |
| Fuel_Type | string | Petrol / Diesel / Electric |
| Seller_Type | string | Dealer |
| Transmission | string | Manual / Automatic |
| Owner | int | Number of previous owners (0 or 1) |

**Dataset stats after outlier removal:** 3,627 rows · 53 car brands · years 1992–2024

---

## Model Summary

### Regression — Polynomial Regression (degree 2)

Chosen because car price has a non-linear relationship with age and mileage. A degree-2 polynomial captures this curve far better than a straight line.

| Metric | Value |
|---|---|
| MAE | 0.0943 Lakhs |
| RMSE | 0.1978 Lakhs |
| R² Score | **0.9999** |
| 5-Fold CV R² | **0.9999** |

### Classification — Decision Tree

Chosen because price categories (Low / Medium / High) have clear threshold boundaries in the data — exactly what decision trees are built to find.

| Metric | Value |
|---|---|
| Accuracy | **99.45%** |
| Macro Precision | 0.9944 |
| Macro Recall | 0.9946 |
| Macro F1 Score | **0.9944** |
| ROC AUC (OVR) | 0.9969 |

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction/car_price_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Transform the dataset
```bash
python transform_used_cars.py
```

### 4. Train the models
```bash
python train_models.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploy on Streamlit Community Cloud

1. Push the project to a **public GitHub repository** (include the `models/` and `assets/` folders).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set:
   - **Main file path:** `car_price_project/app.py`
   - **Branch:** `main`
4. Click **Deploy**.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · joblib · Matplotlib · Seaborn · Streamlit
