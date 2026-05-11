# 🏨 Hotel Booking Cancellation & Revenue Intelligence

A full-stack data science project that analyzes hotel booking data to predict cancellations, forecast revenue, and surface actionable guest insights — complete with an interactive Streamlit web application.

---

## 📌 Project Overview

This project uses the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) to build a production-grade analytics and machine learning pipeline. It covers everything from raw data cleaning to interactive dashboards, including:

- **Cancellation risk prediction** using XGBoost Classifier (F1: 0.71, Accuracy: 0.84)
- **ADR (Average Daily Rate) prediction** using XGBoost Regressor (R²: 0.88, RMSE: 21.5)
- **Booking similarity engine** based on cosine similarity
- **Time-series demand analysis** with STL decomposition
- **Repeated guest loyalty analysis** with statistical hypothesis testing
- **AI Predictor Dashboard** — an end-to-end interactive prediction interface

---

## 👩‍💻 Team & Contributions

| Contributor | Contributions |
|---|---|
| **Esmira** | Feature encoding, EDA & visualizations, time-series analysis, booking similarity engine, repeated guest loyalty analysis, and the Streamlit pages for each of these modules |
| **Zarifa** | Data cleaning pipeline, ML modeling (classification & regression), and the AI Predictor Streamlit dashboard |

---

## 🗂️ Project Structure

```
ProjectFinal/
│
├── hotel_bookings (1) - encoding - viz - stat - modelling.ipynb   # Main notebook
│
├── pages/
│   ├── AIPredictionCenter.py     # 🤖 AI Predictor dashboard (Zarifa)
│   ├── loyalty_analysis.py       # 💚 Repeated guest loyalty analysis (Esmira)
│   └── time_series.py            # 📈 Time-series demand trends (Esmira)
│
├── similarity.py                 # 🔍 Booking similarity engine (Esmira)
│
├── bookings_clean.parquet        # Cleaned & engineered dataset
├── similarity_matrix.npy         # Precomputed cosine similarity matrix
├── similarity_scaler.pkl         # StandardScaler for similarity features
└── similarity_cols.pkl           # Feature columns for similarity model
```

---

## 📓 Notebook Walkthrough

The main notebook is organized into the following sections:

### 1. 🧹 Data Cleaning *(Zarifa)*
- Parsed datetime columns and dropped leaky/irrelevant features (`company`, `reservation_status`)
- Imputed missing values for `agent`, `country`, and `children`
- Removed duplicates and zero-guest records
- Applied IQR-based outlier capping for `adr`, `lead_time`, `adults`, `children`, and stay duration columns

### Statistical Inference *(Zarifa)*
- Hypothesis Testing: Conducted Mann-Whitney U tests to confirm that differences in ADR and lead time between repeat and first-time guests are statistically significant - Categorical Correlation: Applied Chi-squared tests to validate the strong relationship between guest loyalty and lower cancellation rates.
- Segmented Profiling: Analyzed behavioral patterns showing that repeat guests book significantly closer to arrival dates and have a higher density of special requests.- Automated Insights: Engineered a reporting logic to quantify revenue potential,

### 2. 🔢 Feature Encoding *(Esmira)*
- Engineered new features: `total_nights`, `total_guests`, `revenue_potential`, `actual_revenue`, `price_per_person`, `request_density`, `net_bookings`, `room_mismatch`, `is_new_guest`, `deposit_given`, `is_family`, `is_holiday` (Portugal public holidays), `arrival_season`, `waitlist_cat`, `booking_lead_time_bucket`, `revenue_category`
- Applied `LabelEncoder` on binary columns and `OneHotEncoder` (with rare-label grouping) on categorical columns
- Extracted reservation date components (`res_year`, `res_month`, `res_day`, `reservation_day_of_week`)

### 3. 📊 Visualizations & EDA *(Esmira)*
- Cancellation rate by hotel type, deposit type, market segment, customer type, and room type
- Revenue potential vs. actual revenue (Plotly funnel/waterfall)
- ADR distributions by arrival season and hotel type (KDE, violin, box plots)
- Geospatial cancellation map by country
- Lead time vs. cancellation density plots
- Booking changes vs. cancellation rate
- Family vs. non-family revenue comparison

### 4. 🤖 Modeling *(Zarifa)*
- **Classification target:** `is_canceled`
  - SMOTE oversampling (`sampling_strategy=0.8`) to handle class imbalance
  - Compared Random Forest vs. XGBoost; selected XGBoost with threshold 0.35
  - Final metrics: F1 = 0.71, Accuracy = 0.84
- **Regression target:** `adr`
  - Compared Random Forest Regressor vs. XGBoost Regressor
  - Final metrics: R² = 0.88, RMSE ≈ 21.5
- Feature importance analysis for both models

### 5. 🔍 Booking Similarity Engine *(Esmira)*
- Built a cosine similarity matrix over key booking features (`lead_time`, `adr`, `total_nights`, `total_guests`, `special_requests`, `room_mismatch`, `is_family`, `deposit_given`, `customer_type`)
- Serialized matrix, scaler, and feature columns for reuse in the Streamlit app

### 6. 📈 Time-Series Analysis *(Esmira)*
- Monthly confirmed bookings and cancellation rate trends by hotel type and market segment
- STL (Seasonal-Trend decomposition using LOESS) on overall demand
- Cancellation heatmap by year × month

### 7. 💚 Repeated Guest Loyalty Analysis *(Esmira)*
- Compared repeat vs. first-time guests across ADR, lead time, special requests, cancellation rate, and revenue
- Statistical testing: Mann-Whitney U (continuous metrics) and Chi-squared (categorical metrics)
- Room type preferences and stay duration patterns by guest type
- Automated insight report generation

---

## 🖥️ Streamlit App Pages

### 🤖 AI Prediction Center (`AIPredictionCenter.py`) — *Zarifa*
An interactive dashboard with a dark UI (styled for Baku Hotel) featuring three tabs:
- **Prediction tab:** Sidebar inputs (hotel type, lead time, guests, market segment, deposit, etc.) → cancellation probability gauge + predicted ADR metric cards
- **Business Analytics tab:** EDA charts (lead time histogram, market segment cancellation rates, ADR by hotel type, special requests impact)
- **Model Performance tab:** Classifier and regressor metrics, feature importance bar charts

### 📈 Time-Series Demand Trends (`pages/time_series.py`) — *Esmira*
- Interactive filters for hotel type and market segment
- Monthly booking volume and cancellation rate line charts
- STL decomposition (trend + seasonal + residual components)
- Cancellation rate heatmap by year/month

### 💚 Repeated Guest Loyalty Analysis (`pages/loyalty_analysis.py`) — *Esmira*
- Side-by-side comparison of repeat vs. first-time guest profiles
- Statistical significance indicators for each metric
- Visualizations: ADR distributions, cancellation rates, room preferences, stay duration
- Auto-generated loyalty insight summary

### 🔍 Booking Similarity Engine (`similarity.py`) — *Esmira*
- Look up any booking by ID and find the top-N most similar historical bookings
- Custom profile lookup: enter lead time, ADR, nights, guests, hotel type, customer type → find similar past bookings and their outcomes
- Results styled with color-coded cancellation status

---

## ⚙️ Setup & Running

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly statsmodels joblib pyarrow holidays scipy
```

### Run the app

```bash
streamlit run similarity.py
```

> The `pages/` folder is automatically picked up by Streamlit as multi-page app routes.

### Data

The app expects `hotel_bookings.csv` (the raw dataset) in the project root for the time-series and loyalty pages, and `bookings_clean.parquet` for the similarity engine. The `.pkl` and `.npy` artefacts must also be present in the root directory.

---
## 🌐 Live Demo

Streamlit App:
[https://YOUR-APP-LINK.streamlit.app](https://hotel-booking-end-to-end-project-qmeerdf9afbfuixinvwjpe.streamlit.app/)

## 📊 Model Performance Summary

| Model | Task | Key Metric |
|---|---|---|
| XGBoost Classifier | Cancellation prediction | F1: 0.71 · Accuracy: 0.84 |
| XGBoost Regressor | ADR prediction | R²: 0.88 · RMSE: 21.5 |
| Cosine Similarity | Booking recommendation | Top-N retrieval |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data:** pandas, numpy, pyarrow
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Statistics:** scipy (Mann-Whitney U, Chi-squared)
- **Time Series:** statsmodels (STL decomposition)
- **Visualization:** matplotlib, seaborn, plotly
- **Web App:** Streamlit
- **Serialization:** joblib
