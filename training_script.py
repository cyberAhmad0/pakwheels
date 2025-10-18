import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint

# ─── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv("pakwheels_pakistan_automobile_dataset.csv")

# ─── Clean Data ───────────────────────────────────────────────
df.dropna(inplace=True)
df = df[df['price'] != 0]
df = df[df['transmission'] != 'Not Available']

# ─── Split into Two Sets ──────────────────────────────────────
df_low = df[df['price'] <= 10000000]
df_high = df[df['price'] > 10000000]

print(f"\n🚗 Total Cars in Dataset: {len(df)}")
print(f"💰 Cars ≤ 1 Crore: {len(df_low)}")
print(f"💰 Cars > 1 Crore: {len(df_high)}")

# ─── Common Feature Engineering Function ──────────────────────
def prepare_data(df_segment):
    df_segment = df_segment.copy()
    df_segment["title"] = df_segment["title"].str.lower()

    def extract_parts(text):
        words = text.split()
        brand = words[0] if len(words) > 0 else ""
        model_name = words[1] if len(words) > 1 else ""
        variant = ' '.join(words[2:]) if len(words) > 2 else ""
        return pd.Series([brand, model_name, variant])

    df_segment[['brand', 'model_name', 'variant']] = df_segment['title'].apply(extract_parts)

    FEATURES = ['city', 'fuel_type', 'transmission', 'registered',
                'assembly', 'engine_capacity', 'vehicle_age',
                'brand', 'model_name', 'variant', 'model', 'mileage']
    TARGET = 'price'

    X = df_segment[FEATURES]
    y = df_segment[TARGET]

    return X, y

# ─── Common Preprocessing Function ────────────────────────────
def get_pipeline():
    CATEGORICAL_COLS = ['city', 'fuel_type', 'transmission', 'registered',
                        'assembly', 'brand', 'model_name', 'variant']
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
    ], remainder='passthrough')

    model = RandomForestRegressor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline

# ─── Common Training Function ─────────────────────────────────
def train_model(X, y, segment_name, model_filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': randint(5, 30),
        'regressor__min_samples_split': randint(2, 10),
        'regressor__min_samples_leaf': randint(1, 10)
    }

    search = RandomizedSearchCV(get_pipeline(), param_distributions=param_dist,
                                n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\n✅ Model trained for {segment_name}")
    print(f"📉 MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"📈 R² Score: {r2_score(y_test, y_pred):.4f}")

    joblib.dump(best_model, model_filename)
    print(f"💾 Model saved as {model_filename}")

# ─── Train Models ─────────────────────────────────────────────
X_low, y_low = prepare_data(df_low)
train_model(X_low, y_low, "Cars ≤ 1 Crore", "pakwheels_price_predictor_upto_1cr.joblib")

X_high, y_high = prepare_data(df_high)
train_model(X_high, y_high, "Cars > 1 Crore", "pakwheels_price_predictor_above_1cr.joblib")
