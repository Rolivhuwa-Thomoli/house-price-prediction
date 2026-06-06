"""
House Price Prediction - End-to-End ML Pipeline
================================================
A complete machine learning project for predicting house prices.

Author: Rolivhuwa Thomoli
Date: June 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# STEP 1: DATA GENERATION (Simulating real housing dataset)
# ============================================================

def generate_housing_data(n_samples=2160, random_state=42):
    """Generate a realistic housing dataset for demonstration."""
    np.random.seed(random_state)

    data = {
        'bedrooms': np.random.choice([2, 3, 4, 5, 6], n_samples, p=[0.1, 0.35, 0.30, 0.15, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, p=[0.05, 0.10, 0.30, 0.25, 0.15, 0.10, 0.05]),
        'sqft_living': np.random.lognormal(7.5, 0.5, n_samples).astype(int),
        'sqft_lot': np.random.lognormal(8.5, 1.0, n_samples).astype(int),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.25, 0.15, 0.40, 0.10, 0.10]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'view': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.40, 0.25, 0.15, 0.12, 0.08]),
        'condition': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.02, 0.08, 0.35, 0.40, 0.15]),
        'grade': np.random.choice(range(4, 14), n_samples, p=[0.02, 0.05, 0.10, 0.20, 0.25, 0.18, 0.10, 0.05, 0.03, 0.02]),
        'yr_built': np.random.randint(1900, 2024, n_samples),
        'yr_renovated': np.random.choice([0] + list(range(1980, 2024)), n_samples, p=[0.55] + [0.45/44]*44),
        'lat': np.random.uniform(47.1, 47.8, n_samples),
        'long': np.random.uniform(-122.5, -121.5, n_samples),
    }

    df = pd.DataFrame(data)

    # Calculate price based on features with realistic relationships
    base_price = 50000
    price = (base_price
             + df['sqft_living'] * 180
             + df['bedrooms'] * 25000
             + df['bathrooms'] * 35000
             + df['waterfront'] * 250000
             + df['view'] * 40000
             + df['grade'] * 35000
             + df['condition'] * 20000
             + (df['yr_built'] - 1900) * 2000)

    # Add location premium
    price += (df['lat'] - 47.1) * 300000
    price += (df['long'] + 122.5) * -150000

    # Add noise
    price += np.random.normal(0, 50000, n_samples)
    df['price'] = price.clip(100000, 3000000).astype(int)

    return df


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(df):
    """Perform comprehensive exploratory data analysis."""
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Basic info
    print("\n📊 Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print("\n📈 Statistical Summary:")
    print(df.describe())

    print("\n🔍 Missing Values:")
    print(df.isnull().sum())

    print("\n💰 Price Statistics:")
    print(f"Mean: ${df['price'].mean():,.2f}")
    print(f"Median: ${df['price'].median():,.2f}")
    print(f"Min: ${df['price'].min():,.2f}")
    print(f"Max: ${df['price'].max():,.2f}")

    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

    # Price distribution
    axes[0, 0].hist(df['price'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')

    # Living area distribution
    axes[0, 1].hist(df['sqft_living'], bins=50, color='coral', edgecolor='white', alpha=0.8)
    axes[0, 1].set_title('Living Area (sqft) Distribution')
    axes[0, 1].set_xlabel('Square Feet')

    # Bedrooms count
    df['bedrooms'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 2], color='mediumseagreen')
    axes[0, 2].set_title('Bedrooms Count')
    axes[0, 2].set_xlabel('Number of Bedrooms')

    # Grade distribution
    df['grade'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='mediumpurple')
    axes[1, 0].set_title('Grade Distribution')
    axes[1, 0].set_xlabel('Grade')

    # Year built
    axes[1, 1].hist(df['yr_built'], bins=50, color='gold', edgecolor='white', alpha=0.8)
    axes[1, 1].set_title('Year Built Distribution')
    axes[1, 1].set_xlabel('Year')

    # Waterfront
    df['waterfront'].value_counts().plot(kind='bar', ax=axes[1, 2], color=['skyblue', 'navy'])
    axes[1, 2].set_title('Waterfront View')
    axes[1, 2].set_xlabel('Waterfront (0=No, 1=Yes)')

    plt.tight_layout()
    plt.savefig('../images/distributions.png', dpi=150, bbox_inches='tight')
    print("\n✅ Distribution plot saved to images/distributions.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ Correlation heatmap saved to images/correlation_heatmap.png")
    plt.close()

    # Price vs key features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(df['sqft_living'], df['price'], alpha=0.4, color='steelblue', s=20)
    axes[0].set_xlabel('Living Area (sqft)')
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title('Price vs Living Area')

    waterfront_colors = df['waterfront'].map({0: 'coral', 1: 'navy'})
    axes[1].scatter(df['grade'], df['price'], alpha=0.5, c=waterfront_colors, s=20)
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Price vs Grade (Blue = Waterfront)')

    axes[2].scatter(df['lat'], df['long'], c=df['price'], cmap='viridis',
                    alpha=0.5, s=20)
    axes[2].set_xlabel('Latitude')
    axes[2].set_ylabel('Longitude')
    axes[2].set_title('Price by Location')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Price ($)')

    plt.tight_layout()
    plt.savefig('../images/price_relationships.png', dpi=150, bbox_inches='tight')
    print("✅ Price relationships plot saved to images/price_relationships.png")
    plt.close()

    return df


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Create new features to improve model performance."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # Age of house
    df['house_age'] = 2024 - df['yr_built']

    # Years since renovation
    df['years_since_renovation'] = np.where(
        df['yr_renovated'] > 0,
        2024 - df['yr_renovated'],
        0
    )

    # Total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Price per sqft
    df['price_per_sqft'] = df['price'] / df['sqft_living']

    # Living to lot ratio
    df['living_lot_ratio'] = df['sqft_living'] / df['sqft_lot']

    # Has basement (proxy: more than 1 floor)
    df['has_basement'] = (df['floors'] > 1).astype(int)

    # Is renovated
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # Grade category
    df['grade_category'] = pd.cut(df['grade'], bins=[0, 6, 9, 13],
                                   labels=['Low', 'Medium', 'High'])
    df = pd.get_dummies(df, columns=['grade_category'], prefix='grade')

    print("✅ Features created:")
    print("  - house_age")
    print("  - years_since_renovation")
    print("  - total_rooms")
    print("  - price_per_sqft")
    print("  - living_lot_ratio")
    print("  - has_basement")
    print("  - is_renovated")
    print("  - grade_category (one-hot encoded)")

    return df


# ============================================================
# STEP 4: MODEL TRAINING & EVALUATION
# ============================================================

def train_and_evaluate(df):
    """Train multiple models and compare performance."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Prepare features
    feature_cols = [col for col in df.columns
                    if col not in ['price', 'price_per_sqft', 'grade_category_Low',
                                   'grade_category_Medium', 'grade_category_High']]
    # Actually include the grade dummies
    feature_cols = [col for col in df.columns if col not in ['price', 'price_per_sqft']]

    # Ensure only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number]).drop(columns=['price'], errors='ignore')
    y = df['price']

    print(f"\nFeatures used: {list(X.columns)}")
    print(f"Target: price")
    print(f"Samples: {len(X)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = []

    print("\n📊 Model Comparison:")
    print("-" * 55)
    print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
    print("-" * 55)

    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        print(f"{name:<20} ${rmse:>10,.0f} ${mae:>10,.0f} {r2:>7.3f}")

    # Feature importance (Random Forest)
    rf_model = models['Random Forest']
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n🔝 Top 10 Feature Importances (Random Forest):")
    print(importance_df.head(10).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), y='Feature', x='Importance', palette='viridis')
    plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/feature_importance.png', dpi=150, bbox_inches='tight')
    print("\n✅ Feature importance plot saved to images/feature_importance.png")
    plt.close()

    # Actual vs Predicted
    best_model = rf_model
    y_pred_best = best_model.predict(X_test)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_best, alpha=0.5, color='steelblue', s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted Prices (Random Forest)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("✅ Actual vs Predicted plot saved to images/actual_vs_predicted.png")
    plt.close()

    return pd.DataFrame(results), best_model


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute the complete ML pipeline."""
    import os
    os.makedirs('../images', exist_ok=True)
    os.makedirs('../models', exist_ok=True)

    # Generate data
    print("🔄 Generating housing dataset...")
    df = generate_housing_data(n_samples=2160)
    df.to_csv('../data/house_prices.csv', index=False)
    print(f"✅ Dataset saved: {df.shape}")

    # EDA
    df = perform_eda(df)

    # Feature Engineering
    df = engineer_features(df)

    # Train & Evaluate
    results, best_model = train_and_evaluate(df)

    # Save model
    import pickle
    with open('../models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\n💾 Model saved to models/random_forest_model.pkl")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   - data/house_prices.csv")
    print("   - images/distributions.png")
    print("   - images/correlation_heatmap.png")
    print("   - images/price_relationships.png")
    print("   - images/feature_importance.png")
    print("   - images/actual_vs_predicted.png")
    print("   - models/random_forest_model.pkl")

    return results


if __name__ == "__main__":
    results = main()
