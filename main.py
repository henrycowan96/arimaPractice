import pandas as pd
import numpy as np
from pmdarima import auto_arima
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.title('Chocolate Sales Forecasting')
st.write('This app uses an ARIMA model to forecast weekly chocolate sales from `chocolate_sales.csv`.')

# Load the CSV
try:
    data = pd.read_csv('chocolate_sales.csv', parse_dates=['date'], index_col='date')

    if not isinstance(data.index, pd.DatetimeIndex):
        st.error('The "date" column must be parsed as datetime.')
        st.stop()

    if data.shape[0] != 156:
        st.error('The CSV must have exactly 156 rows for 3 years of weekly data.')
        st.stop()

    if 'sales' not in data.columns:
        st.error('The CSV must contain a "sales" column.')
        st.stop()

except FileNotFoundError:
    st.error('File "chocolate_sales.csv" not found.')
    st.stop()

except Exception as e:
    st.error(f'Error loading CSV: {str(e)}')
    st.stop()

# Split data
train = data.iloc[:104]
test = data.iloc[104:]

# Fit the model and forecast
try:
    model = auto_arima(
        train['sales'],
        seasonal=True,
        m=52,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )

    fitted_model = model.fit(train['sales'])

    # Test set predictions
    predictions = fitted_model.predict(n_periods=len(test))
    rmse = np.sqrt(mean_squared_error(test['sales'], predictions))
    st.write(f'RMSE on test set: {rmse:.2f}')

    # Refit on full data
    full_model = auto_arima(
        data['sales'],
        seasonal=True,
        m=52,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    full_fitted_model = full_model.fit(data['sales'])

    # Forecast 10 weeks ahead
    forecast, conf_int = full_fitted_model.predict(n_periods=10, return_conf_int=True, alpha=0.05)
    forecast_series = pd.Series(forecast)

    # User selection
    week = st.slider('Pick a week to forecast (1–10)', 1, 10)
    pred = forecast[week - 1]
    lower, upper = conf_int[week - 1]
    st.write(f'Week {week} forecast: {pred:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])')

    # Prepare forecast plot
    forecast_dates = pd.date_range(start=data.index[-1], periods=11, freq='W')[1:]
    forecast_df = pd.DataFrame({
        'forecast': forecast,
        'lower_ci': conf_int[:, 0],
        'upper_ci': conf_int[:, 1]
    }, index=forecast_dates)

    # Plotting
    st.write('Sales Forecast Plot:')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['sales'], label='Historical Sales', color='blue')
    ax.plot(forecast_df['forecast'], label='Forecast', color='red')
    ax.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'],
                    color='red', alpha=0.3, label='95% Confidence Interval')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f'Model training or forecasting failed: {str(e)}')
