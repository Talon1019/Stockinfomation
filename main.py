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
    # Configure the Streamlit page
    st.set_page_config(page_title="Market Support and Resistance Analyzer", layout="wide")
    st.title("📊 Market Support and Resistance Analyzer")
    
    # Sidebar for user inputs
    st.sidebar.header("User Input")
    
    # Market selection
    market = st.sidebar.selectbox(
        "Select Market",
        options=["Stock", "Crypto", "Forex"]
    )
    
    # Fetch tickers based on selected market
    if market == "Stock":
        ticker_list = get_stock_tickers()
    elif market == "Crypto":
        ticker_list = get_crypto_tickers()
    elif market == "Forex":
        ticker_list = get_forex_tickers()
    else:
        ticker_list = []
    
    if not ticker_list:
        st.error("Failed to fetch the list of tickers for the selected market.")
        return
    
    # Sidebar ticker search (optional)
    ticker_search = st.sidebar.text_input("Search Ticker")
    filtered_tickers = [t for t in ticker_list if ticker_search.upper() in t.upper()] if ticker_search else ticker_list
    
    # User input for ticker symbol with selectbox
    ticker = st.sidebar.selectbox(
        "Select Ticker Symbol",
        options=filtered_tickers,
        index=0
    ).upper()
    
    # User selection of time frame
    time_frame = st.sidebar.selectbox(
        "Select Time Frame",
        ("1mo", "3mo", "6mo", "1y", "5y")
    )
    
    # Sidebar inputs for technical indicators (optional)
    st.sidebar.header("Technical Indicators Settings")
    rsi_period = st.sidebar.number_input("RSI Period", min_value=5, max_value=50, value=14)
    ema_period = st.sidebar.number_input("EMA Period", min_value=50, max_value=300, value=200)
    atr_period = st.sidebar.number_input("ATR Period", min_value=5, max_value=50, value=14)
    
    # Sidebar inputs for strategy parameters
    st.sidebar.header("Strategy Parameters")
    atr_multiplier = st.sidebar.number_input("ATR Multiplier for Risk", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    ema_cross_limit = st.sidebar.number_input("Max EMA Crosses Allowed", min_value=1, max_value=50, value=20, step=1)
    trade_signal_limit = st.sidebar.number_input("Max Trade Signals Allowed", min_value=1, max_value=50, value=20, step=1)
    ema_distance_limit = st.sidebar.slider("Max Price Distance from EMA (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    
    if ticker:
        # Handle tickers with special characters
        ticker_yf = format_ticker_for_yfinance(ticker, market)
        
        # Map time frames to periods and intervals
        period, interval = time_frame_to_params(time_frame)
        
        # Validate period and interval
        if is_valid_period_interval(time_frame, interval):
            # Try to fetch data
            try:
                data = fetch_data(ticker_yf, period, interval)
                if not data.empty:
                    # Identify support and resistance levels (Conservative)
                    support_levels, resistance_levels = identify_support_resistance(
                        data['Close'],
                        prominence=5,       # Increased prominence for conservativeness
                        num_levels=2,       # Limit to top 2 levels each
                        min_distance=10     # Ensure levels are at least 10 data points apart
                    )

                    # Create main tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Graph", "ℹ️ Info", "💡 Recommendations", "ℹ️ Details"])

                    with tab1:
                        # Plot the data
                        plot_market_data(data, support_levels, resistance_levels, ticker, time_frame, market)

                    with tab2:
                        # Display additional information
                        st.subheader("📋 Data Information")
                        st.write(data.describe())

                        # Display support and resistance levels
                        st.subheader("📌 Support Levels")
                        if len(support_levels) > 0:
                            for idx, level in enumerate(support_levels, 1):
                                st.write(f"{idx}. {level:.2f}")
                        else:
                            st.write("No support levels identified.")

                        st.subheader("📍 Resistance Levels")
                        if len(resistance_levels) > 0:
                            for idx, level in enumerate(resistance_levels, 1):
                                st.write(f"{idx}. {level:.2f}")
                        else:
                            st.write("No resistance levels identified.")

                    with tab3:
                        # Display recommendations
                        display_recommendations(
                            data.copy(),
                            ticker,
                            market,
                            rsi_period=rsi_period,
                            ema_period=ema_period,
                            atr_period=atr_period,
                            atr_multiplier=atr_multiplier,
                            ema_cross_limit=ema_cross_limit,
                            trade_signal_limit=trade_signal_limit,
                            ema_distance_limit=ema_distance_limit / 100  # Convert percentage to decimal
                        )

                    with tab4:
                        # Display common details about the asset
                        display_details(ticker_yf, market)

                else:
                    st.error("No data found for the given ticker and time frame.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
        else:
            st.error(
                f"The interval '{interval}' is not valid for the period '{period}'. "
                "Please select a different time frame or adjust the interval."
            )

@st.cache_data(ttl=86400)  # Cache for 1 day
def get_stock_tickers():
    """
    Fetch the list of stock tickers from the Russell 1000 Index.
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
        st.error(f"Error fetching stock tickers: {e}")
        return []

@st.cache_data(ttl=86400)  # Cache for 1 day
def get_crypto_tickers():
    """
    Provide a predefined list of popular cryptocurrency tickers.
    """
    # List can be expanded or fetched from an API like CoinGecko
    crypto_tickers = [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
        "TRX-USD", "UNI-USD", "LINK-USD", "BCH-USD", "ETC-USD",
        "XLM-USD", "FIL-USD", "ATOM-USD", "VET-USD", "EOS-USD",
        "ICP-USD", "THETA-USD", "ALGO-USD", "AAVE-USD", "CRO-USD",
        "NEO-USD", "MIOTA-USD", "XMR-USD", "KSM-USD", "MKR-USD"
    ]
    return crypto_tickers

@st.cache_data(ttl=86400)  # Cache for 1 day
def get_forex_tickers():
    """
    Provide a predefined list of major forex pair tickers.
    """
    forex_tickers = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X",
        "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "AUDJPY=X", "CHFJPY=X", "CADJPY=X", "NZDJPY=X", "EURCAD=X",
        "GBPCHF=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "EURCHF=X"
    ]
    return forex_tickers

def format_ticker_for_yfinance(ticker, market):
    """
    Format the ticker symbol based on the market.
    """
    if market == "Stock":
        # Replace '.' with '-' for stocks like BRK.B
        ticker = ticker.replace('.', '-')
    elif market == "Crypto":
        # No additional formatting needed for crypto tickers in yFinance
        pass
    elif market == "Forex":
        # No additional formatting needed for forex tickers in yFinance
        pass
    return ticker

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(ticker, period, interval):
    """
    Fetch historical market data using yfinance.
    """
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
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

def identify_support_resistance(closing_prices, prominence=5, num_levels=2, min_distance=10):
    """
    Identify support and resistance levels from closing prices.

    Parameters:
    - closing_prices (pd.Series): Series of closing prices.
    - prominence (float): Required prominence of peaks.
    - num_levels (int): Number of support and resistance levels to identify.
    - min_distance (int): Minimum number of data points between peaks.

    Returns:
    - support_levels (np.ndarray): Array of support levels.
    - resistance_levels (np.ndarray): Array of resistance levels.
    """
    # Remove NaN and infinite values
    closing_prices = closing_prices.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure the data type is numeric
    closing_prices = closing_prices.astype(float)

    # Convert closing prices to a NumPy array
    prices = closing_prices.values.flatten()

    # Identify local maxima (resistance levels)
    resistance_indices, properties = find_peaks(prices, prominence=prominence, distance=min_distance)
    resistance_levels = prices[resistance_indices]
    resistance_prominences = properties['prominences']

    # Identify local minima (support levels)
    inverted_prices = -prices
    support_indices, properties = find_peaks(inverted_prices, prominence=prominence, distance=min_distance)
    support_levels = prices[support_indices]
    support_prominences = properties['prominences']

    # Select the most significant resistance levels
    if len(resistance_levels) > 0:
        resistance_df = pd.DataFrame({
            'Level': resistance_levels,
            'Prominence': resistance_prominences
        })
        resistance_df = resistance_df.sort_values(by='Prominence', ascending=False).head(num_levels)
        resistance_levels = resistance_df['Level'].values
    else:
        resistance_levels = np.array([])

    # Select the most significant support levels
    if len(support_levels) > 0:
        support_df = pd.DataFrame({
            'Level': support_levels,
            'Prominence': support_prominences
        })
        support_df = support_df.sort_values(by='Prominence', ascending=False).head(num_levels)
        support_levels = support_df['Level'].values
    else:
        support_levels = np.array([])

    return support_levels, resistance_levels

def calculate_atr(data, period):
    """
    Calculate the Average True Range (ATR) for the given period.
    """
    try:
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift())
        low_close_prev = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        return atr
    except Exception as e:
        st.error(f"Error calculating ATR: {e}")
        return pd.Series([np.nan]*len(data))

def calculate_technical_indicators(data, rsi_period=14, ema_period=200, atr_period=14):
    """
    Calculate MACD, RSI, EMA, and ATR indicators with customizable periods.
    """
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Missing required column: {col}")
            return data  # or handle appropriately

    try:
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

        # ATR
        data['ATR'] = calculate_atr(data, atr_period)

        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return data

def generate_signals(data, overbought=70, oversold=30, atr_multiplier=2, ema_cross_limit=20, trade_signal_limit=20, ema_distance_limit=0.1):
    """
    Generate buy/sell/hold signals based on the 200 DMA MACD Pullback Strategy.

    Parameters:
    - data (pd.DataFrame): Data containing technical indicators.
    - overbought (float): RSI overbought threshold.
    - oversold (float): RSI oversold threshold.
    - atr_multiplier (float): Multiplier for ATR to set risk.
    - ema_cross_limit (int): Maximum number of EMA crosses within the limit to allow trades.
    - trade_signal_limit (int): Maximum number of trade signals within the limit to allow trades.
    - ema_distance_limit (float): Maximum allowed distance between price and EMA to allow trades.

    Returns:
    - signals (dict): Dictionary containing individual indicator signals.
    - final_signal (str): Overall recommendation ('Buy', 'Sell', 'Hold').
    """
    signals = {}
    
    try:
        # Current price and EMA
        current_price = data['Close'].iloc[-1]
        ema = data['EMA'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Check if price is above or below 200 EMA
        price_above_ema = current_price > ema
        price_below_ema = current_price < ema
        
        # MACD Signals
        if len(data) < 2:
            st.warning("Not enough data to generate MACD signals.")
            signals['Strategy'] = 'Hold'
            return signals, 'Hold'
        
        macd_current = data['MACD_Line'].iloc[-1]
        signal_current = data['Signal_Line'].iloc[-1]
        macd_previous = data['MACD_Line'].iloc[-2]
        signal_previous = data['Signal_Line'].iloc[-2]
        
        # ATR-based Risk
        risk = atr_multiplier * atr
        
        # Identify EMA crosses within the last 'ema_cross_limit' candles
        ema_crosses = ((data['Close'].shift(1) < data['EMA'].shift(1)) & (data['Close'] > data['EMA'])) | \
                      ((data['Close'].shift(1) > data['EMA'].shift(1)) & (data['Close'] < data['EMA']))
        recent_ema_crosses = ema_crosses[-ema_cross_limit:].sum()
        
        # Identify trade signals within the last 'trade_signal_limit' candles
        trade_signals = []
        for i in range(-trade_signal_limit, 0):
            if i >= len(data):
                continue  # Prevent index out of bounds
            if (data['Close'].iloc[i] > data['EMA'].iloc[i] and 
                data['MACD_Line'].iloc[i] < 0 and 
                data['MACD_Line'].iloc[i] > data['Signal_Line'].iloc[i]):
                trade_signals.append('Buy')
            elif (data['Close'].iloc[i] < data['EMA'].iloc[i] and 
                  data['MACD_Line'].iloc[i] > 0 and 
                  data['MACD_Line'].iloc[i] < data['Signal_Line'].iloc[i]):
                trade_signals.append('Sell')
        recent_trade_signals = len(trade_signals)
        
        # Distance from EMA
        distance_from_ema = abs(current_price - ema) / ema  # Percentage distance
        
        # Strategy Conditions for Long Position
        long_condition = (
            price_above_ema and
            macd_current > signal_current and
            macd_previous <= signal_previous and
            recent_ema_crosses <= ema_cross_limit and
            recent_trade_signals <= trade_signal_limit and
            distance_from_ema <= ema_distance_limit  # e.g., within 10% of EMA
        )
        
        # Strategy Conditions for Short Position
        short_condition = (
            price_below_ema and
            macd_current < signal_current and
            macd_previous >= signal_previous and
            recent_ema_crosses <= ema_cross_limit and
            recent_trade_signals <= trade_signal_limit and
            distance_from_ema <= ema_distance_limit  # e.g., within 10% of EMA
        )
        
        # Additional Volatility Filter (e.g., avoid high ATR)
        volatility_threshold = data['ATR'].rolling(window=20).mean().iloc[-1]
        high_volatility = data['ATR'].iloc[-1] > volatility_threshold
        
        # Final Signals
        if not high_volatility:
            if long_condition:
                signals['Strategy'] = 'Buy'
            elif short_condition:
                signals['Strategy'] = 'Sell'
            else:
                signals['Strategy'] = 'Hold'
        else:
            signals['Strategy'] = 'Hold'
        
        # Determine final signal
        final_signal = signals['Strategy']
        
        return signals, final_signal
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return signals, 'Hold'

def plot_market_data(data, support_levels, resistance_levels, ticker, time_frame, market):
    """
    Plot the market data with support and resistance levels using candlesticks.
    Enhanced with annotations and additional visual information.
    """
    try:
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
            linewidths=1.0,
            alpha=0.7
        )

        # Customize appearance
        mc = mpf.make_marketcolors(up='g', down='r', wick='inherit', edge='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridaxis='both')

        # Plot the candlestick chart without dpi
        fig, axes = mpf.plot(
            data_plot,
            type='candle',
            style=s,
            volume=True,  # Show volume
            title=f"{ticker.upper()} Candlestick Chart with Support and Resistance Levels ({time_frame})",
            hlines=hlines,
            returnfig=True,
            figsize=(14, 8),  # Manageable figure size
            mav=(50, 200),  # Add moving averages for better trend visualization
            figratio=(16,9),
            figscale=1.2
            # Removed dpi=100
        )
        
        ax_main = axes[0]  # Main candlestick plot
        ax_volume = axes[1]  # Volume plot

        # Annotate only the last (most recent) support and resistance levels
        if len(support_levels) > 0:
            last_support = support_levels[-1]
            ax_main.axhline(last_support, color='darkgreen', linestyle='dotted', linewidth=1)
            ax_main.text(
                0.99, last_support, f' Last Support: {last_support:.2f}',
                color='darkgreen', va='bottom', ha='right', fontsize=9,
                transform=ax_main.get_yaxis_transform(),
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

        if len(resistance_levels) > 0:
            last_resistance = resistance_levels[-1]
            ax_main.axhline(last_resistance, color='darkred', linestyle='dotted', linewidth=1)
            ax_main.text(
                0.99, last_resistance, f' Last Resistance: {last_resistance:.2f}',
                color='darkred', va='bottom', ha='right', fontsize=9,
                transform=ax_main.get_yaxis_transform(),
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

        # Enhance volume bars with color coding based on price movement
        # Note: mplfinance already handles volume coloring, so no need to modify

        # Adjust layout for better spacing
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting market data: {e}")

def display_recommendations(data, ticker, market, rsi_period=14, ema_period=200, atr_period=14, atr_multiplier=2, ema_cross_limit=20, trade_signal_limit=20, ema_distance_limit=0.1):
    """
    Display market movement recommendations using the 200 DMA MACD Pullback Strategy.
    """
    st.header("💡 Recommendations")

    # Calculate technical indicators with user-defined parameters
    data = calculate_technical_indicators(data, rsi_period=rsi_period, ema_period=ema_period, atr_period=atr_period)

    # Generate signals based on the new strategy
    signals, final_signal = generate_signals(
        data.copy(),
        overbought=70,
        oversold=30,
        atr_multiplier=atr_multiplier,
        ema_cross_limit=ema_cross_limit,
        trade_signal_limit=trade_signal_limit,
        ema_distance_limit=ema_distance_limit  # e.g., 0.1 for 10%
    )

    # Display the recommendation
    if final_signal == 'Buy':
        st.success("**Buy** signal based on the 200 DMA MACD Pullback Strategy.")
    elif final_signal == 'Sell':
        st.error("**Sell** signal based on the 200 DMA MACD Pullback Strategy.")
    else:
        st.info("No clear signal. **Hold** position.")

    # Display individual indicator signals
    st.subheader("📈 Indicator Signals")
    for indicator, signal in signals.items():
        if signal == 'Buy':
            st.markdown(f"**{indicator}:** :green[{signal}]")
        elif signal == 'Sell':
            st.markdown(f"**{indicator}:** :red[{signal}]")
        else:
            st.markdown(f"**{indicator}:** :gray[{signal}]")

    # Plot the indicators
    plot_indicators(data, ticker, overbought=70, oversold=30, ema_period=ema_period)

def plot_indicators(data, ticker, overbought=70, oversold=30, ema_period=200):
    """
    Plot the indicators along with the price chart.
    """
    try:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))

        # Price chart with EMA
        ax1.plot(data['Close'], label='Close Price', color='blue')
        ax1.plot(data['EMA'], label=f'EMA {ema_period}', linestyle='--', color='orange')
        ax1.set_title(f"{ticker.upper()} Price Chart with EMA {ema_period}")
        ax1.legend()

        # MACD chart
        ax2.plot(data['MACD_Line'], label='MACD Line', color='green')
        ax2.plot(data['Signal_Line'], label='Signal Line', linestyle='--', color='red')
        ax2.set_title(f"{ticker.upper()} MACD")
        ax2.legend()

        # RSI chart with adjusted thresholds
        ax3.plot(data['RSI'], label='RSI', color='purple')
        ax3.axhline(overbought, color='red', linestyle='--', label='Overbought')
        ax3.axhline(oversold, color='green', linestyle='--', label='Oversold')
        ax3.set_title(f"{ticker.upper()} RSI")
        ax3.legend()

        # ATR chart
        ax4.plot(data['ATR'], label='ATR', color='brown')
        ax4.set_title(f"{ticker.upper()} ATR")
        ax4.legend()

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting indicators: {e}")

def display_details(ticker_yf, market):
    """
    Display common information about the selected asset based on its market type.
    """
    st.header("ℹ️ Details")

    try:
        ticker_obj = yf.Ticker(ticker_yf)
        info = ticker_obj.info
    except Exception as e:
        st.error(f"Error fetching details: {e}")
        return

    if market == "Stock":
        # Display stock-related information
        stock_info = {
            "Name": info.get("shortName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Full Time Employees": info.get("fullTimeEmployees", "N/A"),
            "Market Cap": format_large_number(info.get("marketCap", "N/A")),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A",
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A")
        }

        for key, value in stock_info.items():
            st.write(f"**{key}:** {value}")

    elif market == "Crypto":
        # Display crypto-related information
        # Note: yfinance has limited info for cryptocurrencies
        crypto_info = {
            "Name": info.get("longName", "N/A"),
            "Market Cap": format_large_number(info.get("marketCap", "N/A")),
            "Previous Close": info.get("previousClose", "N/A"),
            "Open": info.get("open", "N/A"),
            "Bid": info.get("bid", "N/A"),
            "Ask": info.get("ask", "N/A"),
            "Volume": format_large_number(info.get("volume", "N/A")),
            "Average Volume": format_large_number(info.get("averageVolume", "N/A")),
            "Beta": info.get("beta", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }

        for key, value in crypto_info.items():
            st.write(f"**{key}:** {value}")

    elif market == "Forex":
        # Display forex-related information
        # yfinance provides minimal info for forex pairs
        forex_info = {
            "Currency Pair": ticker_yf,
            "Bid": info.get("bid", "N/A"),
            "Ask": info.get("ask", "N/A"),
            "Last Price": info.get("previousClose", "N/A"),
            "Volume": format_large_number(info.get("volume", "N/A"))
        }

        for key, value in forex_info.items():
            st.write(f"**{key}:** {value}")
    else:
        st.write("No details available for the selected market.")

def format_large_number(num):
    """
    Format large numbers into a more readable format (e.g., 1,000,000 as 1M).
    """
    if isinstance(num, (int, float)):
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K"
        else:
            return f"{num:.2f}"
    else:
        return str(num)

def time_frame_to_params(time_frame):
    """
    Map time frames to periods and intervals to maintain a consistent number of bars.
    """
    if time_frame == "1mo":
        # For 1 month, use 1-hour intervals over 5 days to get ~30 bars
        # Note: 1 trading day typically has ~6.5 hours (NYSE)
        # 5 days * 6.5 hours ≈ 32.5 bars
        return ("5d", "1h")
    elif time_frame == "3mo":
        return ("3mo", "1d")
    elif time_frame == "6mo":
        return ("6mo", "1d")
    elif time_frame == "1y":
        return ("1y", "1d")
    elif time_frame == "5y":
        return ("5y", "1wk")
    else:
        return ("1d", "1d")

def is_valid_period_interval(time_frame, interval):
    """
    Check if the period and interval combination is valid.
    """
    valid_intervals = {
        '1mo': ['1h'],
        '3mo': ['1d'],
        '6mo': ['1d'],
        '1y': ['1d'],
        '5y': ['1wk'],
    }

    intervals = valid_intervals.get(time_frame, [])
    return interval in intervals

if __name__ == "__main__":
    main()
