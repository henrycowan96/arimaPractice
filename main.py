import pandas as pd
import numpy as np
from pmdarima import auto_arima
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.title('Chocolate Shop Sales Forecasting')
st.write('This app uses an ARIMA model to forecast weekly sales from chocolate_sales.csv.')

# Load the fucking CSV
try:
    data = pd.read_csv('chocolate_sales.csv', parse_dates=['date'], index_col='date')
    if not isinstance(data.index, pd.DatetimeIndex):
        st.error('The "date" column is fucked up. Needs to be datetime.')
        st.stop()
    if data.shape[0] != 156:
        st.error('CSV needs exactly 156 fucking rows.')
        st.stop()
    if 'sales' not in data.columns:
        st.error('CSV needs a "sales" column, asshole.')
        st.stop()
except FileNotFoundError:
    st.error('chocolate_sales.csv is fucking missing.')
    st.stop()
except Exception as e:
    st.error(f'CSV load fucked up: {str(e)}')
    st.stop()

# Split data
train = data.iloc[:104]
test = data.iloc[104:]

# Fit the fucking model on train
try:
    model = auto_arima(train['sales'], seasonal=True, m=52, max_p=2, max_q=2, max_P=1, max_Q=1, trace=True, error_action='ignore', suppress_warnings=True)
    fitted_model = model.fit(train['sales'])
    
    # Test set predictions
    predictions = fitted_model.predict(n_periods=len(test))
    rmse = np.sqrt(mean_squared_error(test['sales'], predictions))
    st.write(f'RMSE on test set: {rmse:.2f}')
    
    # Fit on full data
    full_model = auto_arima(data['sales'], seasonal=True, m=52, max_p=2, max_q=2, max_P=1, max_Q=1, trace=True, error_action='ignore', suppress_warnings=True)
    full_fitted_model = full_model.fit(data['sales'])
    
    # 10-week forecast with confidence intervals
    forecast_dict = full_fitted_model.predict(n_periods=10, return_conf_int=True, alpha=0.05)
    forecast = pd.Series(forecast_dict[0], index=range(10))
    conf_int = forecast_dict[1]
    
    # Week slider
    week = st.slider('Pick a fucking week (1-10)', 1, 10)
    pred_value = forecast.iloc[week-1]
    lower_ci = conf_int[week-1][0]
    upper_ci = conf_int[week-1][1]
    st.write(f'Week {week} prediction: {pred_value:.2f} (95% CI: [{lower_ci:.2f}, {upper_ci:.2f}])')
    
    # Plotting shit
    forecast_dates = pd.date_range(start=data.index[-1], periods=11, freq='W')[1:]
    forecast_df = pd.DataFrame({
        'forecast': forecast,
        'lower_ci': conf_int[:, 0],
        'upper_ci': conf_int[:, 1]
    }, index=forecast_dates)
    
    st.write('Sales and Forecast Plot:')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['sales'], label='Historical Sales', color='blue')
    ax.plot(forecast_df['forecast'], label='Forecast', color='red')
    ax.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'], color='red', alpha=0.3, label='95% CI')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f'Model or forecast fucked up: {str(e)}')
