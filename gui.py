import streamlit as st
import pandas as pd
import joblib
import os

# ─── Load Models ─────────────────────────────────────────────────────
try:
    model_upto_1cr = joblib.load("pakwheels_price_predictor_upto_1cr.joblib")
    model_above_1cr = joblib.load("pakwheels_price_predictor_above_1cr.joblib")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ─── Load Data for Dropdowns ─────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pakwheels_pakistan_automobile_dataset.csv")
        df.dropna(inplace=True)
        df = df[df['price'] != 0]
        df = df[df['transmission'] != 'Not Available']
        df['title'] = df['title'].str.lower()

        def extract_parts(text):
            words = text.split()
            brand = words[0] if len(words) > 0 else ""
            model_name = words[1] if len(words) > 1 else ""
            variant = ' '.join(words[2:]) if len(words) > 2 else ""
            return pd.Series([brand, model_name, variant])

        df[['brand', 'model_name', 'variant']] = df['title'].apply(extract_parts)
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ─── Unique Dropdowns ────────────────────────────────────────────────
brands = sorted(df['brand'].unique())
models = sorted(df['model_name'].unique())
variants = sorted(df['variant'].unique())
cities = sorted(df['city'].unique())
fuel_types = sorted(df['fuel_type'].unique())
transmissions = sorted(df['transmission'].unique())
registered_in = sorted(df['registered'].unique())
assemblies = sorted(df['assembly'].unique())
years = sorted(df['model'].unique())

# ─── GUI ─────────────────────────────────────────────────────────────
st.title("🚗 PakWheels Car Price Predictor (Simple)")
st.markdown("Select car features to predict price.")

# Ask user if price is above 1 crore
above_1cr = st.radio("Is the car's price expected to be above 1 crore?", ("No", "Yes"))

# Input Features
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands)
    model_name = st.selectbox("Model Name", models)
    variant = st.selectbox("Variant", variants)
    year = st.selectbox("Model Year", years)
    mileage = st.number_input("Mileage (in km)", min_value=0)

with col2:
    city = st.selectbox("City", cities)
    fuel = st.selectbox("Fuel Type", fuel_types)
    transmission = st.selectbox("Transmission", transmissions)
    registered = st.selectbox("Registered In", registered_in)
    assembly = st.selectbox("Assembly", assemblies)
    engine = st.number_input("Engine Capacity (cc)", min_value=600, max_value=8000, step=100)

# Compute vehicle age
vehicle_age = 2025 - int(year)

# ─── Predict Button ──────────────────────────────────────────────────
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([{
            'brand': brand,
            'model_name': model_name,
            'variant': variant,
            'model': year,
            'mileage': mileage,
            'city': city,
            'fuel_type': fuel,
            'transmission': transmission,
            'registered': registered,
            'assembly': assembly,
            'engine_capacity': engine,
            'vehicle_age': vehicle_age
        }])

        model = model_above_1cr if above_1cr == "Yes" else model_upto_1cr
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Price: Rs. {int(prediction):,}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
