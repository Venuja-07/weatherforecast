import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import streamlit as st
import tensorflow as tf # Import tensorflow

@st.cache_data
def fetch_and_prepare_data(lat, lon, start_date, end_date, timezone):
    # Daily temperature data
    daily_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_min,temperature_2m_max,temperature_2m_mean"
        f"&timezone={timezone.replace('/', '%2F')}"
    )
    daily_response = requests.get(daily_url)
    daily_response.raise_for_status()
    daily_data = pd.DataFrame(daily_response.json()["daily"])
    daily_data.rename(columns={"time": "Date", "temperature_2m_mean": "Day Temp (°C)"}, inplace=True)
    daily_data["Date"] = pd.to_datetime(daily_data["Date"])

    # Hourly humidity data
    hourly_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=relative_humidity_2m"
        f"&timezone={timezone.replace('/', '%2F')}"
    )
    hourly_response = requests.get(hourly_url)
    hourly_response.raise_for_status()
    hourly_data = pd.DataFrame(hourly_response.json()["hourly"])
    hourly_data["time"] = pd.to_datetime(hourly_data["time"])
    hourly_data["Date"] = hourly_data["time"].dt.date
    humidity_daily = hourly_data.groupby("Date")["relative_humidity_2m"].mean().reset_index()
    humidity_daily.rename(columns={"relative_humidity_2m": "Humidity (%)"}, inplace=True)
    humidity_daily["Date"] = pd.to_datetime(humidity_daily["Date"])

    # Merge both datasets
    df = pd.merge(daily_data, humidity_daily, on="Date", how="inner")
    df.sort_values("Date", inplace=True)
    return df

# Parameters
lat = -37.8136
lon = 144.9631
start_date = "2023-06-01"
end_date = "2024-06-01"
timezone = "Australia/Sydney"

df = fetch_and_prepare_data(lat, lon, start_date, end_date, timezone)

# Feature engineering
df['DayIndex'] = np.arange(len(df))
df['DayofWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Temp_Lag1'] = df['Day Temp (°C)'].shift(1)
df['Temp_Lag1'] = df['Temp_Lag1'].fillna(df['Temp_Lag1'].iloc[1])


# --- Linear Regression ---
@st.cache_resource
def train_linear_regression(dataframe):
    X_lr = dataframe[['DayIndex', 'Humidity (%)']]
    y_lr = dataframe['Day Temp (°C)']
    lr_model = LinearRegression()
    lr_model.fit(X_lr, y_lr)
    y_lr_pred = lr_model.predict(X_lr)
    return lr_model, y_lr_pred

lr_model, y_lr_pred = train_linear_regression(df)

# --- Prophet ---
@st.cache_resource
def train_prophet(dataframe):
    prophet_df = dataframe[['Date', 'Day Temp (°C)']].rename(columns={'Date': 'ds', 'y': 'y'})
    n_changepoints = min(25, len(prophet_df) - 1)
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                            n_changepoints=n_changepoints)
    prophet_model.fit(prophet_df)
    return prophet_model, prophet_df

prophet_model, prophet_df = train_prophet(df)
future_df_prophet = prophet_model.make_future_dataframe(periods=0) # Initially no future periods for historical fit
prophet_forecast = prophet_model.predict(future_df_prophet)


# --- LSTM ---
@st.cache_resource
def train_lstm(dataframe, look_back=7):
    data_lstm = dataframe['Day Temp (°C)'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_lstm)

    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_lstm, y_lstm = create_dataset(scaled_data, look_back)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_lstm, y_lstm, epochs=100, batch_size=32, verbose=0)

    y_lstm_pred_scaled = model_lstm.predict(X_lstm)
    y_lstm_pred = scaler.inverse_transform(y_lstm_pred_scaled)
    y_lstm_actual = scaler.inverse_transform(y_lstm.reshape(-1, 1))

    return model_lstm, scaler, look_back, X_lstm, y_lstm, y_lstm_pred, y_lstm_actual

model_lstm, scaler, look_back, X_lstm, y_lstm, y_lstm_pred, y_lstm_actual = train_lstm(df)

# --- Evaluation ---
actual = df['Day Temp (°C)'].values
mae_lr = mean_absolute_error(actual, y_lr_pred)
rmse_lr = np.sqrt(mean_squared_error(actual, y_lr_pred))
r2_lr = r2_score(actual, y_lr_pred)

prophet_pred = prophet_forecast['yhat'][:len(actual)]
mae_prophet = mean_absolute_error(actual, prophet_pred)
rmse_prophet = np.sqrt(mean_squared_error(actual, prophet_pred))

# Adjust actual values for LSTM evaluation due to look_back
actual_lstm = actual[look_back:]
mae_lstm = mean_absolute_error(actual_lstm, y_lstm_pred)
rmse_lstm = np.sqrt(mean_squared_error(actual_lstm, y_lstm_pred))


# --- Streamlit App Layout ---
st.title("Weather Forecasting Dashboard")

st.header("Model Evaluation Metrics (Historical Data)")
st.write(f"🔹 Linear Regression:  MAE = {mae_lr:.2f}, RMSE = {rmse_lr:.2f}, R² = {r2_lr:.2f}")
st.write(f"🔹 Prophet:            MAE = {mae_prophet:.2f}, RMSE = {rmse_prophet:.2f}")
st.write(f"🔹 LSTM:               MAE = {mae_lstm:.2f}, RMSE = {rmse_lstm:.2f}")


st.header("Historical Temperature Forecasts")

# Plot historical forecasts
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], actual, label='Actual Temp', marker='o', linewidth=1)
ax.plot(df['Date'], y_lr_pred, label='Linear Regression', linestyle='--')
ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Forecast', linestyle='--')
ax.plot(df['Date'].iloc[look_back:], y_lstm_pred, label='LSTM Forecast', linestyle='--') # Align LSTM dates

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Melbourne Historical Temperature Forecasts")
ax.legend()
ax.grid(True)
st.pyplot(fig)


st.header("Future Temperature Forecasts")

# Interactive widget for future forecast days
future_periods = st.number_input("Number of future days to forecast:", min_value=1, max_value=365, value=30)

# --- Prophet Future Forecast ---
future_df_prophet = prophet_model.make_future_dataframe(periods=future_periods)
prophet_future_forecast = prophet_model.predict(future_df_prophet)

# --- LSTM Future Forecast ---
# Create the input sequence for forecasting the first future day
last_look_back_days = scaler.transform(df['Day Temp (°C)'].values.reshape(-1, 1))[-look_back:]
future_forecast_input_lstm = last_look_back_days.reshape(1, look_back, 1)

# List to store future predictions
lstm_future_predictions_scaled = []

# Predict future values in a loop
for _ in range(future_periods):
    # Predict the next day's temperature
    next_day_prediction_scaled = model_lstm.predict(future_forecast_input_lstm, verbose=0)
    lstm_future_predictions_scaled.append(next_day_prediction_scaled[0, 0])

    # Update the input sequence for the next prediction
    future_forecast_input_lstm = np.append(future_forecast_input_lstm[:, 1:, :], next_day_prediction_scaled.reshape(1, 1, 1), axis=1)

# Inverse transform the scaled future predictions
lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions_scaled).reshape(-1, 1))

# Generate future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods, freq='D')

# Create a DataFrame for LSTM future forecasts
lstm_future_forecast_df = pd.DataFrame({
    'ds': future_dates,
    'yhat': lstm_future_predictions.flatten()
})


# Plot historical and future forecasts
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(df['Date'], actual, label='Actual Temp', marker='o', linewidth=1)
ax2.plot(df['Date'], y_lr_pred, label='Linear Regression (Historical)', linestyle='--')
ax2.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Forecast (Historical)', linestyle='--')
ax2.plot(df['Date'].iloc[look_back:], y_lstm_pred, label='LSTM Forecast (Historical)', linestyle='--') # Align LSTM dates

# Plot future forecasts
ax2.plot(prophet_future_forecast['ds'], prophet_future_forecast['yhat'], label='Prophet Forecast (Future)', linestyle='-')
ax2.plot(lstm_future_forecast_df['ds'], lstm_future_forecast_df['yhat'], label='LSTM Forecast (Future)', linestyle='-')


ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title("Melbourne Temperature Forecasts (Historical and Future)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)


st.header("How to Run the App")
st.write("1. Save the code above as `weather_dashboard.py`")
st.write("2. Open your terminal or command prompt.")
st.write("3. Navigate to the directory where you saved the file.")
st.write("4. Run the command: `streamlit run weather_dashboard.py`")
st.write("This will open the dashboard in your web browser.")
