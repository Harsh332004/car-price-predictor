import streamlit as st
import pickle
import pandas as pd

# Page Title
st.title("Car Price Predictor")

# Load the dataset
car = pickle.load(open('car.pkl', 'rb'))

# Load the trained model (Pipeline)
model = pickle.load(open('LinearRegression.pkl', 'rb'))

# Dropdown inputs
car_name = st.selectbox("Select Car Name", sorted(car['name'].unique()))
company = st.selectbox("Select Company", sorted(car['company'].unique()))
year = st.number_input("Enter Year of Purchase", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Select Fuel Type", sorted(car['fuel_type'].unique()))

# Predict Button
if st.button("Predict"):
    # Prepare input DataFrame
    input_df = pd.DataFrame([[car_name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Predict using loaded model
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Price of the car is ₹{int(prediction):,}")
