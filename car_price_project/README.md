# Second-Hand Car Price Prediction & Classification

A machine-learning course project that predicts the **selling price** of used cars (regression) and classifies the price into **Low / Medium / High** categories (classification), deployed as an interactive Streamlit web app.

---

## 📁 Project Structure

```
car_price_project/
├── data/
│   └── car_data.csv              # Raw dataset (302 rows, 9 columns)
├── notebooks/
│   └── car_price.ipynb           # Full EDA + training notebook (4 Units)
├── src/
│   ├── __init__.py
│   └── preprocessing.py          # Custom sklearn transformers & helpers
├── models/
│   ├── regressor.joblib          # Best regression pipeline
│   ├── classifier.joblib         # Best classification pipeline
│   └── metadata.json             # Dropdown options, ranges, metrics
├── assets/                       # Saved EDA and evaluation charts
├── train_models.py               # Standalone training script
├── app.py                        # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Column | Type | Description |
|---|---|---|
| Car_Name | string | Car model name (brand extracted) |
| Year | int | Year of manufacture |
| Selling_Price | float | Selling price in Lakhs ₹ (target) |
| Present_Price | float | Current showroom price in Lakhs ₹ |
| Kms_Driven | int | Odometer reading |
| Fuel_Type | string | Petrol / Diesel / CNG |
| Seller_Type | string | Dealer / Individual |
| Transmission | string | Manual / Automatic |
| Owner | int | Number of previous owners |

---

## ⚙️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction/car_price_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (generates models/ and assets/)
```bash
python train_models.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

> **Important:** The `.joblib` files are in `.gitignore` by default.  
> To deploy, you must either:
> - **Option A (recommended):** Remove `models/*.joblib` and `assets/*.png` from `.gitignore`, run `train_models.py` locally, commit all generated files, then push.
> - **Option B:** Add a startup script that runs `train_models.py` before the app starts.

### Steps
1. Push your project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo → set:
   - **Main file path:** `car_price_project/app.py`
   - **Branch:** `main`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

---

## 🧪 Model Summary

### Regression (predicts Selling_Price in Lakhs ₹)
| Model | MAE (₹ L) | RMSE (₹ L) | Test R² | 5-Fold CV R² |
|---|---|---|---|---|
| Simple Linear (Present_Price) | 1.019 | 1.465 | 0.752 | 0.735 |
| Multiple Linear | 0.764 | 1.090 | 0.863 | 0.915 |
| Polynomial (deg=2) | 1.592 | 2.982 | -0.028 | -80.523 |
| Ridge (α=1) | 0.759 | 1.050 | 0.872 | 0.931 |
| Lasso (α=0.01) | 0.768 | 1.048 | 0.873 | 0.925 |
| ElasticNet | 0.850 | 1.080 | 0.865 | 0.873 |
| **Random Forest Reg ✓** | **0.445** | **0.870** | **0.912** | **0.946** |

### Classification (predicts Low / Medium / High)
| Model | Accuracy | Macro F1 | ROC-AUC (OVR) |
|---|---|---|---|
| Logistic Regression | 87.5% | 0.873 | 0.969 |
| KNN (k=5) | 83.9% | 0.834 | 0.951 |
| **Decision Tree ✓** | **91.1%** | **0.909** | **0.933** |
| Random Forest Clf | 85.7% | 0.852 | 0.960 |
| SVM (RBF) | 85.7% | 0.853 | 0.953 |
| Gaussian NB | 76.8% | 0.744 | 0.900 |

---

## 📝 Limitations

- Dataset is small (302 rows); predictions may be noisy for rare car models.
- `Present_Price` (showroom price) is a strong predictor — in a real deployment you may not always have it.
- The model was trained on Indian market data; results may not generalise to other markets.
- The `CURRENT_YEAR` constant is fixed at 2024 in `src/preprocessing.py` for reproducibility.

---

## 🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · joblib · Matplotlib · Seaborn · Streamlit
