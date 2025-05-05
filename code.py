import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('📈 Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("⚠️ Education Purpose only")

scaler = StandardScaler()
data = pd.DataFrame()

def main():
    global data
    # Data source: Upload or Download
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain 'Date' and 'Close')", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=['Date'])
        data.set_index('Date', inplace=True)
        st.sidebar.success("✅ CSV uploaded successfully")
    else:
        option = st.sidebar.text_input('Enter a Stock Symbol', value='INFY').upper()
        today = datetime.date.today()
        duration = st.sidebar.number_input('Enter duration (days)', value=3000)
        before = today - datetime.timedelta(days=duration)
        start_date = st.sidebar.date_input('Start Date', value=before)
        end_date = st.sidebar.date_input('End date', value=today)

        if start_date >= end_date:
            st.sidebar.error('❌ End date must be after start date')
            return

        st.sidebar.success(f'Start: {start_date}, End: {end_date}')
        data = download_data(option, start_date, end_date)

    # Navigation
    choice = st.sidebar.selectbox('Choose Action', ['Visualize', 'Recent Data', 'Predict'])
    if not data.empty:
        if choice == 'Visualize':
            tech_indicators()
        elif choice == 'Recent Data':
            dataframe()
        else:
            predict()
    else:
        st.warning("⚠️ No data loaded. Please upload a CSV or enter a valid stock symbol.")


@st.cache_resource
def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return df


def tech_indicators():
    st.header('📊 Technical Indicators')
    indicator = st.radio('Select Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Prepare indicators
    bb_indicator = BollingerBands(data.Close)
    data['bb_h'] = bb_indicator.bollinger_hband()
    data['bb_l'] = bb_indicator.bollinger_lband()

    macd = MACD(data.Close).macd()
    rsi = RSIIndicator(data.Close).rsi()
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    ema = EMAIndicator(data.Close).ema_indicator()

    if indicator == 'Close':
        st.line_chart(data['Close'])
    elif indicator == 'BB':
        st.line_chart(data[['Close', 'bb_h', 'bb_l']])
    elif indicator == 'MACD':
        st.line_chart(macd)
    elif indicator == 'RSI':
        st.line_chart(rsi)
    elif indicator == 'SMA':
        st.line_chart(sma)
    elif indicator == 'EMA':
        st.line_chart(ema)


def dataframe():
    st.header('📄 Recent Data')
    st.dataframe(data.tail(10))


def predict():
    st.header('📈 Predict Future Stock Prices')
    model_name = st.radio('Select Model', [
        'LinearRegression', 'RandomForestRegressor',
        'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'
    ])
    num = int(st.number_input('Forecast how many days?', min_value=1, max_value=100, value=5))

    if st.button('Predict'):
        if model_name == 'LinearRegression':
            model_engine(LinearRegression(), num)
        elif model_name == 'RandomForestRegressor':
            model_engine(RandomForestRegressor(), num)
        elif model_name == 'ExtraTreesRegressor':
            model_engine(ExtraTreesRegressor(), num)
        elif model_name == 'KNeighborsRegressor':
            model_engine(KNeighborsRegressor(), num)
        else:
            model_engine(XGBRegressor(), num)


def model_engine(model, num):
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)

    X = df[['Close']].values
    X_scaled = scaler.fit_transform(X)
    X_forecast = X_scaled[-num:]
    X_scaled = X_scaled[:-num]
    y = df['preds'].dropna().values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    st.write(f"**R² Score**: {r2_score(y_test, preds):.4f}")
    st.write(f"**MAE**: {mean_absolute_error(y_test, preds):.4f}")

    # Forecast
    st.subheader(f"📅 Forecast for next {num} day(s):")
    forecast = model.predict(X_forecast)
    for i, price in enumerate(forecast, 1):
        st.text(f"Day {i}: {price:.2f}")


if __name__ == '__main__':
    main()
