import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Title of the app
st.title("🚗 Car Fuel Efficiency - Regression Dashboard")

# Load dataset
try:
    df = pd.read_csv("car_fuel_efficiency.csv")
    st.header("📊 Dataset Preview")
    st.write(df.head())

    st.header("📈 Feature Distributions")
    st.line_chart(df)
except FileNotFoundError:
    st.error("❌ Dataset file 'car_fuel_efficiency.csv' not found.")
    st.stop()

# Show model metrics
st.header("📋 Model Comparison Results")
if os.path.exists("car_metrics.txt"):
    with open("car_metrics.txt", "r") as f:
        st.text(f.read())
else:
    st.warning("⚠️ Metrics file 'car_metrics.txt' not found. Please generate it.")

# Show loss curve if available
st.header("📉 Neural Network Loss Over Epochs")
if os.path.exists("car_loss_plot.png"):
    st.image("car_loss_plot.png")
else:
    st.warning("⚠️ Loss plot 'car_loss_plot.png' not found. Please run the training script to generate it.")

# Footer
st.markdown("---")
st.markdown("📄 Made for **Neural Network Assignment** – using an alternate regression dataset.")
