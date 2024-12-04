import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

def main():
    st.title("Stock Support and Resistance Analyzer")

    # Fetch the list of top 1000 common stocks (Russell 1000 Index)
    ticker_list = get_russell_1000_tickers()

    if not ticker_list:
        st.error("Failed to fetch the list of top 1000 stocks.")
        return

    # User input for ticker symbol with selectbox
    ticker = st.selectbox(
        "Enter Stock Ticker Symbol",
        options=ticker_list,
        index=ticker_list.index('AAPL') if 'AAPL' in ticker_list else 0
    ).upper()

    # User selection of time frame
    time_frame = st.selectbox(
        "Select Time Frame",
        ("1mo", "3mo", "6mo", "1y", "5y")
    )

    if ticker:
        # Handle tickers with special characters
        ticker_yf = format_ticker_for_yfinance(ticker)

        # Map time frames to intervals
        interval = time_frame_to_interval(time_frame)

        # Validate period and interval
        if is_valid_period_interval(time_frame, interval):
            # Try to fetch data
            try:
                data = fetch_data(ticker_yf, time_frame, interval)
                if not data.empty:
                    # Identify support and resistance levels
                    support_levels, resistance_levels = identify_support_resistance(data['Close'])

                    # Create tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Graph", "ℹ️ Info", "💡 Recommendations", "🔄 Backtesting"])

                    with tab1:
                        # Plot the data
                        plot_stock_data(data, support_levels, resistance_levels, ticker, time_frame)

                    with tab2:
                        # Display additional information
                        st.write("### Data Information")
                        st.write(data.describe())

                        # Display support and resistance levels
                        st.write("### Support Levels")
                        st.write(support_levels)
                        st.write("### Resistance Levels")
                        st.write(resistance_levels)

                    with tab3:
                        # Display recommendations
                        display_recommendations(data.copy(), ticker)

                    with tab4:
                        # Backtesting
                        perform_backtesting(ticker_yf, ticker)

                else:
                    st.error("No data found for the given ticker and time frame.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
        else:
            st.error(
                f"The interval '{interval}' is not valid for the period '{time_frame}'. "
                "Please select a different time frame or adjust the interval."
            )

@st.cache_data
def get_russell_1000_tickers():
    """
    Fetch the list of tickers in the Russell 1000 Index from Wikipedia.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table with the list of tickers
        table = soup.find('table', {'class': 'wikitable sortable'})

        # Extract the tickers from the table
        ticker_list = []
        for row in table.findAll('tr')[1:]:
            cols = row.findAll('td')
            if len(cols) >= 2:
                ticker = cols[1].text.strip()
                # Handle tickers with periods (e.g., BRK.B)
                ticker = ticker.replace('.', '-')
                ticker_list.append(ticker)

        return ticker_list
    except Exception as e:
        st.error(f"Error fetching Russell 1000 tickers: {e}")
        return []

def format_ticker_for_yfinance(ticker):
    """
    Format the ticker symbol to match yfinance requirements.
    """
    # Replace '.' with '-' (e.g., BRK.B becomes BRK-B)
    ticker = ticker.replace('.', '-')
    return ticker

@st.cache_data
def fetch_data(ticker, time_frame, interval):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        data = yf.download(
            tickers=ticker,
            period=time_frame,
            interval=interval,
            group_by='ticker',  # Keep default
            auto_adjust=False,
            prepost=False,
            threads=True,
            proxy=None
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    if data.empty:
        st.error(
            f"No data found for ticker '{ticker}'. Please check if the ticker symbol is correct."
        )
        return pd.DataFrame()

    # Handle multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten the MultiIndex columns
        data.columns = [' '.join(col).strip() for col in data.columns.values]

    # Remove ticker symbol from column names if present at the beginning
    data.columns = [col.replace(f'{ticker} ', '') for col in data.columns]

    # Ensure all required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"Data is missing required columns: {missing_cols}")
        st.write("Columns in fetched data:")
        st.write(list(data.columns))
        return pd.DataFrame()

    return data

def identify_support_resistance(closing_prices):
    """
    Identify support and resistance levels from closing prices.
    """
    # Remove NaN and infinite values
    closing_prices = closing_prices.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure the data type is numeric
    closing_prices = closing_prices.astype(float)

    # Convert closing prices to a NumPy array
    prices = closing_prices.values.flatten()

    # Identify local maxima (resistance levels)
    resistance_indices, _ = find_peaks(prices, prominence=1)
    resistance_levels = prices[resistance_indices]

    # Identify local minima (support levels)
    inverted_prices = -prices
    support_indices, _ = find_peaks(inverted_prices, prominence=1)
    support_levels = prices[support_indices]

    # Select the most significant levels
    support_levels = select_significant_levels(support_levels)
    resistance_levels = select_significant_levels(resistance_levels)

    return support_levels, resistance_levels

def select_significant_levels(levels, num_levels=3):
    """
    Select the most significant levels.
    """
    levels = np.sort(levels)
    if len(levels) > num_levels:
        idx = np.linspace(0, len(levels) - 1, num_levels).astype(int)
        levels = levels[idx]
    return levels

def plot_stock_data(data, support_levels, resistance_levels, ticker, time_frame):
    """
    Plot the stock data with support and resistance levels using candlesticks.
    """
    # Select and clean data for mplfinance
    data_plot = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Drop rows with NaN values
    data_plot.dropna(inplace=True)

    # Ensure data types are correct
    data_plot[['Open', 'High', 'Low', 'Close', 'Volume']] = data_plot[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # Create horizontal lines for support and resistance
    hlines = dict(
        hlines=list(support_levels) + list(resistance_levels),
        colors=['green'] * len(support_levels) + ['red'] * len(resistance_levels),
        linestyle='dashed',
        linewidths=1.5,
        alpha=0.7
    )

    # Customize appearance
    mc = mpf.make_marketcolors(up='g', down='r', wick='inherit', edge='inherit')
    s = mpf.make_mpf_style(marketcolors=mc)

    # Plot the candlestick chart
    fig, _ = mpf.plot(
        data_plot,
        type='candle',
        style=s,
        volume=False,
        title=f"{ticker.upper()} Candlestick Chart with Support and Resistance Levels ({time_frame})",
        hlines=hlines,
        returnfig=True,
        figsize=(12, 6)
    )
    st.pyplot(fig)

def display_recommendations(data, ticker):
    """
    Display stock movement recommendations using RSI, MACD, and EMA.
    """
    st.header("Recommendation")

    # Calculate technical indicators with default parameters
    data = calculate_technical_indicators(data, rsi_period=14, ema_period=200)

    # Generate signals based on indicators with default thresholds
    signals, final_signal = generate_signals(data, overbought=70, oversold=30)

    # Display the recommendation
    if final_signal == 'Buy':
        st.success("All indicators suggest a **Buy** signal.")
    elif final_signal == 'Sell':
        st.error("All indicators suggest a **Sell** signal.")
    else:
        st.info("The indicators are not in agreement. No clear signal.")

    # Display individual indicator signals
    st.write("### Indicator Signals")
    for indicator, signal in signals.items():
        st.write(f"**{indicator}**: {signal}")

    # Plot the indicators
    plot_indicators(data, ticker, overbought=70, oversold=30, ema_period=200)

def calculate_technical_indicators(data, rsi_period=14, ema_period=200):
    """
    Calculate MACD, RSI, and EMA indicators with customizable periods.
    """
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = data['Close'].diff(1)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=rsi_period, min_periods=1).mean()
    loss = down.rolling(window=rsi_period, min_periods=1).mean()
    RS = gain / loss
    data['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    return data

def generate_signals(data, overbought=70, oversold=30):
    """
    Generate buy/sell signals based on MACD, RSI, and EMA with customizable thresholds.
    """
    signals = {}

    # RSI Signal
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi < oversold:
        signals['RSI'] = 'Buy'
    elif latest_rsi > overbought:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Hold'

    # MACD Signal
    if data['MACD_Line'].iloc[-1] > data['Signal_Line'].iloc[-1]:
        signals['MACD'] = 'Buy'
    elif data['MACD_Line'].iloc[-1] < data['Signal_Line'].iloc[-1]:
        signals['MACD'] = 'Sell'
    else:
        signals['MACD'] = 'Hold'

    # EMA Signal
    current_price = data['Close'].iloc[-1]
    ema = data['EMA'].iloc[-1]
    if current_price > ema:
        signals['EMA'] = 'Buy'
    else:
        signals['EMA'] = 'Sell'

    # Determine final signal
    if all(signal == 'Buy' for signal in signals.values()):
        final_signal = 'Buy'
    elif all(signal == 'Sell' for signal in signals.values()):
        final_signal = 'Sell'
    else:
        final_signal = 'Hold'

    return signals, final_signal

def plot_indicators(data, ticker, overbought=70, oversold=30, ema_period=200):
    """
    Plot the indicators along with the price chart.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Price chart with EMA
    ax1.plot(data['Close'], label='Close Price')
    ax1.plot(data['EMA'], label=f'EMA {ema_period}', linestyle='--')
    ax1.set_title(f"{ticker.upper()} Price Chart with EMA {ema_period}")
    ax1.legend()

    # MACD chart
    ax2.plot(data['MACD_Line'], label='MACD Line')
    ax2.plot(data['Signal_Line'], label='Signal Line', linestyle='--')
    ax2.set_title(f"{ticker.upper()} MACD")
    ax2.legend()

    # RSI chart with adjusted thresholds
    ax3.plot(data['RSI'], label='RSI')
    ax3.axhline(overbought, color='red', linestyle='--')
    ax3.axhline(oversold, color='green', linestyle='--')
    ax3.set_title(f"{ticker.upper()} RSI")
    ax3.legend()

    plt.tight_layout()
    st.pyplot(fig)

def perform_backtesting(ticker_yf, ticker_display):
    st.header("Backtesting Results")

    # Fetch data specifically for backtesting (1 year, daily interval)
    backtest_time_frame = '1y'
    backtest_interval = '1d'

    data = fetch_data(ticker_yf, backtest_time_frame, backtest_interval)

    if data.empty:
        st.error("No data available for backtesting.")
        return

    # Calculate indicators with specified parameters
    data = calculate_technical_indicators(data, rsi_period=14, ema_period=200)

    # Create tabs for each strategy
    rsi_tab, macd_tab, ema_tab, all_tab = st.tabs([
        "RSI Strategy", "MACD Strategy", "EMA Strategy", "Combined Strategy"
    ])

    with rsi_tab:
        backtest_rsi_strategy(data.copy(), ticker_display)

    with macd_tab:
        backtest_macd_strategy(data.copy(), ticker_display)

    with ema_tab:
        backtest_ema_strategy(data.copy(), ticker_display)

    with all_tab:
        positions = backtest_all_strategy(data.copy(), ticker_display)
        if positions:
            trade_analysis(positions)
        else:
            st.write("No trades to analyze.")

def backtest_rsi_strategy(data, ticker):
    st.subheader("Backtesting: RSI Strategy (Long Positions Only)")

    # Parameters
    overbought = 70
    oversold = 30
    initial_balance = 10000
    position = 0  # 1 if holding a position, 0 if not
    data['Position'] = 0  # Position status
    data['Buy_Price'] = np.nan
    data['Sell_Price'] = np.nan

    for i in range(len(data)):
        if position == 0:
            if data['RSI'].iloc[i] < oversold:
                position = 1
                data['Position'].iloc[i] = 1
                data['Buy_Price'].iloc[i] = data['Close'].iloc[i]
        elif position == 1:
            if data['RSI'].iloc[i] > overbought:
                position = 0
                data['Position'].iloc[i] = 0
                data['Sell_Price'].iloc[i] = data['Close'].iloc[i]
            else:
                data['Position'].iloc[i] = 1
        else:
            data['Position'].iloc[i] = position

    # Forward-fill positions
    data['Position'] = data['Position'].fillna(method='ffill')

    # Calculate returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    # Plot results
    plot_backtesting_results(data, ticker, "RSI Strategy")

def backtest_macd_strategy(data, ticker):
    st.subheader("Backtesting: MACD Strategy (Long Positions Only)")

    initial_balance = 10000
    position = 0
    data['Position'] = 0
    data['Buy_Price'] = np.nan
    data['Sell_Price'] = np.nan

    for i in range(1, len(data)):
        if position == 0:
            if data['MACD_Line'].iloc[i] > data['Signal_Line'].iloc[i] and \
               data['MACD_Line'].iloc[i - 1] <= data['Signal_Line'].iloc[i - 1]:
                position = 1
                data['Position'].iloc[i] = 1
                data['Buy_Price'].iloc[i] = data['Close'].iloc[i]
            else:
                data['Position'].iloc[i] = 0
        elif position == 1:
            if data['MACD_Line'].iloc[i] < data['Signal_Line'].iloc[i] and \
               data['MACD_Line'].iloc[i - 1] >= data['Signal_Line'].iloc[i - 1]:
                position = 0
                data['Position'].iloc[i] = 0
                data['Sell_Price'].iloc[i] = data['Close'].iloc[i]
            else:
                data['Position'].iloc[i] = 1

    # Forward-fill positions
    data['Position'] = data['Position'].fillna(method='ffill')

    # Calculate returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    # Plot results
    plot_backtesting_results(data, ticker, "MACD Strategy")

def backtest_ema_strategy(data, ticker):
    st.subheader("Backtesting: EMA Strategy (Long Positions Only)")

    initial_balance = 10000
    position = 0
    data['Position'] = 0
    data['Buy_Price'] = np.nan
    data['Sell_Price'] = np.nan

    for i in range(1, len(data)):
        if position == 0:
            if data['Close'].iloc[i] > data['EMA'].iloc[i] and \
               data['Close'].iloc[i - 1] <= data['EMA'].iloc[i - 1]:
                position = 1
                data['Position'].iloc[i] = 1
                data['Buy_Price'].iloc[i] = data['Close'].iloc[i]
            else:
                data['Position'].iloc[i] = 0
        elif position == 1:
            if data['Close'].iloc[i] < data['EMA'].iloc[i] and \
               data['Close'].iloc[i - 1] >= data['EMA'].iloc[i - 1]:
                position = 0
                data['Position'].iloc[i] = 0
                data['Sell_Price'].iloc[i] = data['Close'].iloc[i]
            else:
                data['Position'].iloc[i] = 1

    # Forward-fill positions
    data['Position'] = data['Position'].fillna(method='ffill')

    # Calculate returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    # Plot results
    plot_backtesting_results(data, ticker, "EMA Strategy")

def backtest_all_strategy(data, ticker):
    st.subheader("Backtesting: Combined Strategy (Long Positions Only)")

    # Parameters
    overbought = 70
    oversold = 30
    initial_balance = 10000
    position = 0
    data['Position'] = 0
    data['Buy_Price'] = np.nan
    data['Sell_Price'] = np.nan

    positions = []

    for i in range(1, len(data)):
        # Entry Conditions
        if position == 0:
            if (data['RSI'].iloc[i] < oversold) and \
               (data['MACD_Line'].iloc[i] > data['Signal_Line'].iloc[i]) and \
               (data['Close'].iloc[i] > data['EMA'].iloc[i]):
                position = 1
                data['Position'].iloc[i] = 1
                data['Buy_Price'].iloc[i] = data['Close'].iloc[i]
                entry_info = {
                    'Entry Date': data.index[i],
                    'Entry Price': data['Close'].iloc[i]
                }
        # Exit Conditions
        elif position == 1:
            if (data['RSI'].iloc[i] > overbought) or \
               (data['MACD_Line'].iloc[i] < data['Signal_Line'].iloc[i]) or \
               (data['Close'].iloc[i] < data['EMA'].iloc[i]):
                position = 0
                data['Position'].iloc[i] = 0
                data['Sell_Price'].iloc[i] = data['Close'].iloc[i]
                exit_info = {
                    'Exit Date': data.index[i],
                    'Exit Price': data['Close'].iloc[i]
                }
                trade_info = {**entry_info, **exit_info}
                positions.append(trade_info)
            else:
                data['Position'].iloc[i] = 1

    # Forward-fill positions
    data['Position'] = data['Position'].fillna(method='ffill')

    # Calculate returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    # Plot results
    plot_backtesting_results(data, ticker, "Combined Strategy")

    return positions

def plot_backtesting_results(data, ticker, strategy_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot cumulative returns
    ax1.plot(data.index, data['Cumulative_Market_Returns'], label='Market Returns', color='blue')
    ax1.plot(data.index, data['Cumulative_Strategy_Returns'], label='Strategy Returns', color='orange')
    ax1.set_title(f"Cumulative Returns: {strategy_name} on {ticker}")
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend()

    # Plot buy and sell signals on price chart
    ax2.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
    ax2.scatter(data.index, data['Buy_Price'], label='Buy Signal', marker='^', color='green', s=100)
    ax2.scatter(data.index, data['Sell_Price'], label='Sell Signal', marker='v', color='red', s=100)
    ax2.set_title(f"{ticker} Price Chart with Buy/Sell Signals ({strategy_name})")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

def trade_analysis(positions):
    st.subheader("Trade Analysis for Combined Strategy")

    positions_df = pd.DataFrame(positions)

    if positions_df.empty:
        st.write("No trades to analyze.")
        return

    st.write(positions_df)

def time_frame_to_interval(time_frame):
    """
    Map time frames to data intervals.
    """
    if time_frame == "1mo":
        return "1d"    # Daily interval for 1 month
    elif time_frame == "3mo":
        return "1d"    # Daily interval for 3 months
    elif time_frame == "6mo":
        return "1d"    # Daily interval for 6 months
    elif time_frame == "1y":
        return "1d"    # Daily interval for 1 year
    elif time_frame == "5y":
        return "1wk"   # Weekly interval for 5 years
    else:
        return "1d"

def is_valid_period_interval(period, interval):
    """
    Check if the period and interval combination is valid.
    """
    valid_intervals = {
        '1mo': ['1d'],
        '3mo': ['1d'],
        '6mo': ['1d'],
        '1y': ['1d'],
        '5y': ['1wk'],
    }

    intervals = valid_intervals.get(period, [])
    return interval in intervals

if __name__ == "__main__":
    main()
