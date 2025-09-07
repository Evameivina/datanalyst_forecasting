# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Load & Preprocess Data
# =========================
@st.cache_data
def load_data():
    url = "http://raw.githubusercontent.com/Evameivina/datanalyst_forecasting/refs/heads/main/forecasting_dataset.csv"
    df = pd.read_csv(url)

    # Konversi kolom Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Feature engineering tanggal
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    return df

df = load_data()

# =========================
# 2. Sidebar Filter
# =========================
st.sidebar.header("Filter Data")
year_filter = st.sidebar.multiselect("Pilih Tahun", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
month_filter = st.sidebar.multiselect("Pilih Bulan", options=sorted(df["Month"].unique()), default=sorted(df["Month"].unique()))
irradiance_range = st.sidebar.slider("Rentang Irradiance (kWh/m²)", float(df["Irradiance (kWh/m²)"].min()), float(df["Irradiance (kWh/m²)"].max()), (float(df["Irradiance (kWh/m²)"].min()), float(df["Irradiance (kWh/m²)"].max())))

df_filtered = df[
    (df["Year"].isin(year_filter)) &
    (df["Month"].isin(month_filter)) &
    (df["Irradiance (kWh/m²)"] >= irradiance_range[0]) &
    (df["Irradiance (kWh/m²)"] <= irradiance_range[1])
]

# =========================
# 3. Exploratory Analysis
# =========================
st.title("☀️ Solar Output Prediction Dashboard")
st.write("Dashboard ini menampilkan analisis dan prediksi Solar Output berdasarkan regresi linear.")

# Ringkasan data
st.subheader("Ringkasan Data")
st.dataframe(df_filtered.head())

# Korelasi
st.subheader("Matriks Korelasi")
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
sns.heatmap(df_filtered.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Tren Solar Output
st.subheader("Tren Solar Output per Waktu")
fig_trend, ax_trend = plt.subplots(figsize=(8, 4))
ax_trend.plot(df_filtered["Date"], df_filtered["Solar Output (MWh)"], marker="o", linestyle="-")
ax_trend.set_xlabel("Tanggal")
ax_trend.set_ylabel("Solar Output (MWh)")
ax_trend.set_title("Tren Solar Output")
st.pyplot(fig_trend)

# =========================
# 4. Model & Prediction
# =========================
st.subheader("Prediksi Solar Output dengan Linear Regression")

# Fitur & Target
X = df_filtered[["Irradiance (kWh/m²)", "Temperature (°C)", "Humidity (%)", "Year", "Month", "Day", "Weekday"]]
y = df_filtered["Solar Output (MWh)"]

# Train-Test Split
if len(df_filtered) > 10:  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col3.metric("R²", f"{r2:.2f}")

    # Plot Actual vs Predicted
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(y_test, y_pred, alpha=0.7)
    ax_pred.set_xlabel("Actual")
    ax_pred.set_ylabel("Predicted")
    ax_pred.set_title("Actual vs Predicted Solar Output")
    st.pyplot(fig_pred)

    # =========================
    # 5. Prediksi Custom Input
    # =========================
    st.subheader("Prediksi Berdasarkan Input Manual")
    col_a, col_b, col_c = st.columns(3)
    irr_input = col_a.number_input("Irradiance (kWh/m²)", min_value=0.0, value=float(df["Irradiance (kWh/m²)"].mean()))
    temp_input = col_b.number_input("Temperature (°C)", min_value=0.0, value=float(df["Temperature (°C)"].mean()))
    hum_input = col_c.number_input("Humidity (%)", min_value=0.0, value=float(df["Humidity (%)"].mean()))

    date_input = st.date_input("Pilih Tanggal", value=pd.to_datetime(df["Date"].iloc[0]))
    year_val = date_input.year
    month_val = date_input.month
    day_val = date_input.day
    weekday_val = date_input.weekday()

    if st.button("Prediksi Solar Output"):
        input_data = np.array([[irr_input, temp_input, hum_input, year_val, month_val, day_val, weekday_val]])
        pred_output = model.predict(input_data)[0]
        st.success(f"Prediksi Solar Output: {pred_output:.3f} MWh")
else:
    st.warning("Data terlalu sedikit untuk melatih model.")
