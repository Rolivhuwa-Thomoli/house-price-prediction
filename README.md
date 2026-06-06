# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices based on various features like location, size, bedrooms, and amenities. This project demonstrates the complete ML pipeline from data exploration to model deployment.

---

## 📌 Problem Statement

Predicting house prices accurately is crucial for real estate companies, buyers, and sellers. This project builds a regression model to estimate house prices using property features, helping stakeholders make data-driven decisions.

## 🎯 Key Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Engineer meaningful features to improve model performance
- Compare multiple regression algorithms
- Evaluate model performance using appropriate metrics
- Create actionable insights from data

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Scikit-Learn** — Machine learning models and preprocessing
- **Matplotlib & Seaborn** — Data visualization
- **XGBoost** — Gradient boosting for improved predictions

## 📊 Dataset

The dataset contains **2,160 house records** with the following features:

| Feature | Description |
|---------|-------------|
| `price` | Sale price (target variable) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Living area in square feet |
| `sqft_lot` | Lot size in square feet |
| `floors` | Number of floors |
| `waterfront` | Has waterfront view (0/1) |
| `view` | Quality of view (0-4) |
| `condition` | Overall condition (1-5) |
| `grade` | Construction grade (1-13) |
| `yr_built` | Year built |
| `yr_renovated` | Year renovated |
| `lat` / `long` | Geographic coordinates |

## 📈 Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | $175,432 | 0.712 |
| Random Forest | $142,891 | 0.809 |
| **XGBoost** | **$128,654** | **0.845** |
| Gradient Boosting | $131,223 | 0.838 |

**Best Model:** XGBoost Regressor with 84.5% R² score

## 🔑 Key Insights

- **Square footage** is the strongest predictor of house price
- Houses with **waterfront views** command a 45% premium on average
- **Location (latitude/longitude)** significantly impacts pricing
- Properties built after **2000** show 23% higher average prices
- **Grade and condition** together explain 60% of price variance

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
# Navigate to the project directory
cd house-price-prediction

# Run the Jupyter notebook
jupyter notebook src/house_price_prediction.ipynb

# Or run the Python script
python src/house_price_prediction.py
```

## 📁 Project Structure

```
house-price-prediction/
├── data/
│   └── house_prices.csv
├── src/
│   ├── house_price_prediction.ipynb    # Full analysis notebook
│   └── house_price_prediction.py       # Production-ready script
├── models/
│   └── xgboost_model.pkl               # Saved best model
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
├── requirements.txt
└── README.md
```

## 🎓 What I Learned

- Feature engineering techniques: log transformation, polynomial features, binning
- Handling skewed distributions in real-world datasets
- Hyperparameter tuning with RandomizedSearchCV
- Interpreting model results using SHAP values
- The importance of cross-validation for robust evaluation

---

**Status:** ✅ Completed | **Last Updated:** June 2026
