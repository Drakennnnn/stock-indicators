import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import finnhub
import requests
import time
from datetime import datetime, timedelta, date
import pytz
import websocket
import json
import threading
import queue
from io import StringIO
import csv
import pickle
import os
from functools import partial

# Set page configuration
st.set_page_config(
    page_title="Advanced Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API keys
FINNHUB_API_KEY = "d03bkkpr01qvvb93ems0d03bkkpr01qvvb93emsg"  # For real-time data
ALPHA_VANTAGE_API_KEY = "1DCSD59WWFZD4A6A"  # For historical data

# Data persistence functions
def save_paper_trading_data():
    """Save paper trading data to disk"""
    try:
        data_to_save = {
            'portfolio': st.session_state.paper_portfolio,
            'last_signals': st.session_state.last_signals if 'last_signals' in st.session_state else [],
            'signal_performance': st.session_state.signal_performance if 'signal_performance' in st.session_state else {}
        }
        
        with open('paper_trading_data.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)
        
        st.success("Paper trading data saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving paper trading data: {e}")
        return False

def load_paper_trading_data():
    """Load paper trading data from disk"""
    try:
        if os.path.exists('paper_trading_data.pkl'):
            with open('paper_trading_data.pkl', 'rb') as f:
                data = pickle.load(f)
                st.session_state.paper_portfolio = data['portfolio']
                if 'last_signals' in data:
                    st.session_state.last_signals = data['last_signals']
                else:
                    st.session_state.last_signals = []
                    
                # Load signal performance tracking data
                if 'signal_performance' in data:
                    st.session_state.signal_performance = data['signal_performance']
                else:
                    st.session_state.signal_performance = {}
                    
            st.sidebar.success("Loaded saved trading data!")
            return True
        return False
    except Exception as e:
        st.error(f"Error loading paper trading data: {e}")
        return False

# Function to save API keys to disk
def save_api_keys():
    """Save API keys to disk"""
    try:
        keys_to_save = {
            'alpha_vantage': ALPHA_VANTAGE_API_KEY,
            'finnhub': FINNHUB_API_KEY
        }
        
        with open('api_keys.json', 'w') as f:
            json.dump(keys_to_save, f)
        return True
    except Exception as e:
        st.error(f"Error saving API keys: {e}")
        return False

# Function to load API keys from disk
def load_api_keys():
    """Load API keys from disk"""
    global ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY
    
    try:
        if os.path.exists('api_keys.json'):
            with open('api_keys.json', 'r') as f:
                keys = json.load(f)
                if 'alpha_vantage' in keys:
                    ALPHA_VANTAGE_API_KEY = keys['alpha_vantage']
                if 'finnhub' in keys:
                    FINNHUB_API_KEY = keys['finnhub']
            return True
        return False
    except Exception as e:
        st.error(f"Error loading API keys: {e}")
        return False

# Load API keys if available
load_api_keys()

# Initialize Finnhub client for real-time data
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Message queue for websocket data
ws_message_queue = queue.Queue()

# Default settings
DEFAULT_LOOKBACK_DAYS = {
    "short": 30,    # Short-term (swing trading)
    "medium": 90,   # Medium-term
    "long": 200     # Long-term
}

# Trading timeframe definitions
TIMEFRAMES = {
    "short": "Short-term (1-7 days)",
    "medium": "Medium-term (1-4 weeks)",
    "long": "Long-term (1-3 months)"
}

# Multiple timeframe settings for analysis
MULTI_TIMEFRAME_SETTINGS = {
    "short": ["15min", "1h", "1d"],  # For short-term signals
    "medium": ["1h", "1d", "1wk"],   # For medium-term signals
    "long": ["1d", "1wk", "1mon"]    # For long-term signals
}

# Initialize session state for stock universe if not exists
if 'stock_universe' not in st.session_state:
    st.session_state.stock_universe = None
    
# Initialize paper trading session state
def initialize_paper_trading():
    if 'paper_portfolio' not in st.session_state:
        # Try to load saved data first
        if not load_paper_trading_data():
            # If loading fails, initialize with defaults
            st.session_state.paper_portfolio = {
                'cash': 100000,  # Starting with $100k
                'positions': {},  # Will store {symbol: {'shares': qty, 'entry_price': price, 'entry_time': time, 'signal_confidence': conf}}
                'trade_history': [],  # Will store closed trades
                'performance': {'total_return': 0, 'win_rate': 0, 'trades': 0}
            }
    
    if 'last_signals' not in st.session_state:
        st.session_state.last_signals = []  # Store last scanned signals to avoid duplicates
        
    # Initialize signal performance tracking
    if 'signal_performance' not in st.session_state:
        st.session_state.signal_performance = {
            'signals': {},  # Will store {signal_key: {'wins': n, 'losses': n, 'total': n, 'win_rate': x}}
            'indicators': {},  # Will store performance by indicator
            'patterns': {}  # Will store performance by pattern combination
        }

# ENHANCEMENT 5: Signal Performance Tracking Functions
def update_signal_performance(signal_key, outcome):
    """
    Update the historical performance of a signal type
    
    Parameters:
    signal_key (str): A unique identifier for this signal type
    outcome (bool): True for win, False for loss
    """
    if 'signal_performance' not in st.session_state:
        initialize_paper_trading()
    
    performance = st.session_state.signal_performance
    
    # Initialize if signal key doesn't exist
    if signal_key not in performance['signals']:
        performance['signals'][signal_key] = {
            'wins': 0,
            'losses': 0,
            'total': 0,
            'win_rate': 0.0
        }
    
    # Update counts
    if outcome:
        performance['signals'][signal_key]['wins'] += 1
    else:
        performance['signals'][signal_key]['losses'] += 1
        
    performance['signals'][signal_key]['total'] += 1
    
    # Update win rate
    total = performance['signals'][signal_key]['total']
    wins = performance['signals'][signal_key]['wins']
    performance['signals'][signal_key]['win_rate'] = (wins / total) * 100 if total > 0 else 0
    
    # Save updated data
    save_paper_trading_data()
    
def generate_signal_key(signal):
    """Generate a unique key for a signal based on its characteristics"""
    # Create a key based on direction and reasons
    if not signal or 'direction' not in signal or 'reasons' not in signal:
        return None
        
    # Sort reasons to ensure consistent key generation
    sorted_reasons = sorted(signal['reasons'])
    reason_str = "_".join([r.replace(" ", "")[:15] for r in sorted_reasons])
    
    # Include market regime if available
    regime_str = f"_{signal.get('market_regime', 'unknown')}"
    
    # Create the key
    key = f"{signal['direction']}_{reason_str}{regime_str}"
    
    return key

def get_signal_historical_performance(signal):
    """Get the historical performance metrics for a given signal type"""
    signal_key = generate_signal_key(signal)
    
    if not signal_key or 'signal_performance' not in st.session_state:
        return None
    
    performance = st.session_state.signal_performance
    
    if signal_key in performance['signals']:
        return performance['signals'][signal_key]
    
    return None

# Initialize websocket connection for real-time data
def init_websocket():
    def on_message(ws, message):
        data = json.loads(message)
        if data['type'] == 'trade':
            for trade in data['data']:
                ws_message_queue.put(trade)
    
    def on_error(ws, error):
        print(f"Error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("### Websocket closed ###")
        time.sleep(5)  # Wait before trying to reconnect
        init_websocket()
    
    def on_open(ws):
        print("Websocket connection opened")
        symbols = st.session_state.get('ws_symbols', [])
        for symbol in symbols:
            ws.send(f'{{"type":"subscribe","symbol":"{symbol}"}}')
    
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.on_open = on_open
    
    # Start the websocket in a separate thread
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    return ws

# Function to test API keys
def verify_api_keys():
    av_valid = False
    fh_valid = False
    
    # Test Alpha Vantage API key
    try:
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        if "Global Quote" in data:
            st.sidebar.success("✅ Alpha Vantage API connected successfully")
            av_valid = True
        else:
            st.sidebar.error("❌ Alpha Vantage API returned unexpected response")
            # Offer option to generate new key
            if st.sidebar.button("Generate New Alpha Vantage API Key"):
                new_key = generate_alpha_vantage_key()
                if new_key:
                    st.sidebar.success(f"Generated new Alpha Vantage API key: {new_key[:5]}...")
                    # We don't need to restart - the global variables are updated directly
    except Exception as e:
        st.sidebar.error(f"❌ Alpha Vantage API connection failed: {e}")
        # Offer option to generate new key
        if st.sidebar.button("Generate New Alpha Vantage API Key"):
            new_key = generate_alpha_vantage_key()
            if new_key:
                st.sidebar.success(f"Generated new Alpha Vantage API key: {new_key[:5]}...")

    # Test Finnhub API key
    try:
        quote = finnhub_client.quote('AAPL')
        if 'c' in quote:
            st.sidebar.success("✅ Finnhub API connected successfully")
            fh_valid = True
        else:
            st.sidebar.error("❌ Finnhub API returned unexpected response")
    except Exception as e:
        st.sidebar.error(f"❌ Finnhub API connection failed: {e}")
    
    return av_valid, fh_valid

# Function to get a comprehensive list of US stocks
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_extended_stock_universe():
    try:
        # First try to get an expanded stock list from Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200 and response.content:
            # Parse the CSV content
            csv_data = StringIO(response.text)
            reader = csv.reader(csv_data)
            next(reader)  # Skip header row
            
            # Filter for active US stocks only
            us_stocks = []
            for row in reader:
                if len(row) >= 3 and row[2] == 'stock' and row[3] == 'United States':
                    # Store both symbol and name
                    us_stocks.append((row[0], row[1]))
            
            if len(us_stocks) > 0:
                return us_stocks
        
        # Fallback to basic list if API fails
        return get_basic_stock_universe()
            
    except Exception as e:
        st.warning(f"Could not fetch extended stock list: {e}. Using default list instead.")
        return get_basic_stock_universe()

# Basic list of major US stocks as fallback
def get_basic_stock_universe():
    basic_stocks = [
        ("AAPL", "Apple Inc"),
        ("MSFT", "Microsoft Corporation"),
        ("AMZN", "Amazon.com Inc"),
        ("GOOGL", "Alphabet Inc Class A"),
        ("META", "Meta Platforms Inc"),
        ("TSLA", "Tesla Inc"),
        ("NVDA", "NVIDIA Corporation"),
        ("JPM", "JPMorgan Chase & Co"),
        ("BAC", "Bank of America Corp"),
        ("V", "Visa Inc"),
        ("JNJ", "Johnson & Johnson"),
        ("PG", "Procter & Gamble Co"),
        ("UNH", "UnitedHealth Group Inc"),
        ("HD", "Home Depot Inc"),
        ("XOM", "Exxon Mobil Corp"),
        ("DIS", "Walt Disney Co"),
        ("NFLX", "Netflix Inc"),
        ("INTC", "Intel Corporation"),
        ("AMD", "Advanced Micro Devices Inc"),
        ("PYPL", "PayPal Holdings Inc"),
        ("CSCO", "Cisco Systems Inc"),
        ("VZ", "Verizon Communications Inc"),
        ("T", "AT&T Inc"),
        ("CMCSA", "Comcast Corporation"),
        ("KO", "Coca-Cola Co"),
        ("PEP", "PepsiCo Inc"),
        ("WMT", "Walmart Inc"),
        ("MCD", "McDonald's Corp"),
        ("NKE", "Nike Inc"),
        ("SBUX", "Starbucks Corporation")
    ]
    return basic_stocks

# Load stock universe on startup
def load_stock_universe():
    if st.session_state.stock_universe is None:
        with st.spinner("Loading US stock universe..."):
            st.session_state.stock_universe = get_extended_stock_universe()
    return st.session_state.stock_universe

# ENHANCEMENT 1: Multi-timeframe analysis functions
def fetch_multi_timeframe_data(symbol, timeframe_preference="short"):
    """
    Fetch data for multiple timeframes based on the preferred trading timeframe
    
    Parameters:
    symbol (str): The stock symbol
    timeframe_preference (str): One of "short", "medium", or "long"
    
    Returns:
    dict: A dictionary with data for each timeframe
    """
    # Get the appropriate timeframes to fetch based on preference
    timeframes = MULTI_TIMEFRAME_SETTINGS[timeframe_preference]
    
    # Initialize results dictionary
    multi_tf_data = {}
    
    # Fetch data for each timeframe
    for tf in timeframes:
        if tf == "15min" or tf == "1h":
            # Fetch intraday data for shorter timeframes
            interval = "15min" if tf == "15min" else "60min"
            data = fetch_alpha_vantage_intraday(symbol, interval=interval)
            if not data.empty:
                multi_tf_data[tf] = data
        elif tf == "1d":
            # Use the existing daily data fetch
            data = create_comprehensive_dataset(symbol, "short")
            if not data.empty:
                multi_tf_data[tf] = data
        elif tf == "1wk":
            # Weekly data
            data = fetch_alpha_vantage_weekly(symbol)
            if not data.empty:
                multi_tf_data[tf] = data
        elif tf == "1mon":
            # Monthly data
            data = fetch_alpha_vantage_monthly(symbol)
            if not data.empty:
                multi_tf_data[tf] = data
    
    return multi_tf_data

# Function to fetch weekly data from Alpha Vantage
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_alpha_vantage_weekly(symbol):
    try:
        # Alpha Vantage API call for weekly data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        if "Weekly Time Series" not in data:
            st.warning(f"No weekly data found for {symbol} in Alpha Vantage. Error: {data.get('Information', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["Weekly Time Series"]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Convert columns to numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Keep only the last 52 weeks (1 year)
        if len(df) > 52:
            df = df.iloc[-52:]
        
        # Reset index and rename it to 'timestamp' to maintain compatibility
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Add data source column
        df['data_source'] = 'weekly'
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching weekly data for {symbol} from Alpha Vantage: {e}")
        return pd.DataFrame()

# Function to fetch monthly data from Alpha Vantage
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_alpha_vantage_monthly(symbol):
    try:
        # Alpha Vantage API call for monthly data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        if "Monthly Time Series" not in data:
            st.warning(f"No monthly data found for {symbol} in Alpha Vantage. Error: {data.get('Information', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["Monthly Time Series"]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Convert columns to numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Keep only the last 24 months (2 years)
        if len(df) > 24:
            df = df.iloc[-24:]
        
        # Reset index and rename it to 'timestamp' to maintain compatibility
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Add data source column
        df['data_source'] = 'monthly'
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching monthly data for {symbol} from Alpha Vantage: {e}")
        return pd.DataFrame()

# Function to fetch historical daily data from Alpha Vantage
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_alpha_vantage_daily(symbol, timeframe="short"):
    try:
        # Determine days back based on timeframe
        days_back = DEFAULT_LOOKBACK_DAYS[timeframe]
        
        # Determine output size (compact for short-term, full for longer timeframes)
        output_size = "compact" if timeframe == "short" else "full"
        
        # Alpha Vantage API call for daily data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            st.warning(f"No data found for {symbol} in Alpha Vantage. Error: {data.get('Information', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Convert columns to numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Keep only the required days
        if days_back and len(df) > days_back:
            df = df.iloc[-days_back:]
        
        # Reset index and rename it to 'timestamp' to maintain compatibility
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Add data source column
        df['data_source'] = 'daily'
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching daily data for {symbol} from Alpha Vantage: {e}")
        return pd.DataFrame()

# Fetch intraday data from Alpha Vantage for the current trading day
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_alpha_vantage_intraday(symbol, interval='15min'):
    """
    Fetch intraday data from Alpha Vantage
    
    Parameters:
    symbol (str): The stock symbol
    interval (str): Time interval between data points - 1min, 5min, 15min, 30min, 60min
    
    Returns:
    pd.DataFrame: DataFrame with intraday OHLCV data
    """
    try:
        # Make API call to Alpha Vantage for intraday data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        # Check if data was returned successfully
        if f"Time Series ({interval})" not in data:
            st.warning(f"No intraday data found for {symbol} in Alpha Vantage. Error: {data.get('Information', 'Unknown error')}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data[f"Time Series ({interval})"]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Convert columns to numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # For 15min and 1h, get a reasonable amount of history
        lookback_days = 5 if interval == '15min' else 10 if interval == '60min' else 1
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df.index >= cutoff_date]
        
        # Reset index and rename it to 'timestamp' to maintain compatibility
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Add data source column
        df['data_source'] = f'intraday_{interval}'
        
        return df
        
    except Exception as e:
        st.warning(f"Error fetching intraday data for {symbol} from Alpha Vantage: {e}")
        return pd.DataFrame()

# Function to get current quote from Finnhub (real-time)
def get_current_quote(symbol):
    try:
        quote = finnhub_client.quote(symbol)
        return quote
    except Exception as e:
        st.warning(f"Error fetching current quote for {symbol}: {e}")
        return None

# Create a comprehensive dataset combining daily, intraday, and real-time data
def create_comprehensive_dataset(symbol, timeframe="short"):
    """
    Create a comprehensive dataset combining historical daily data, 
    intraday data for the current day, and the latest real-time quote.
    
    Parameters:
    symbol (str): The stock symbol
    timeframe (str): The timeframe for historical data ("short", "medium", "long")
    
    Returns:
    pd.DataFrame: A combined DataFrame with all available data
    """
    # Step 1: Get historical daily data from Alpha Vantage
    daily_df = fetch_alpha_vantage_daily(symbol, timeframe)
    if daily_df.empty:
        st.error(f"Could not fetch historical daily data for {symbol}")
        return pd.DataFrame()
    
    # Check if we already have today's data in daily dataset
    today = datetime.now().date()
    today_in_daily = any(d.date() == today for d in daily_df['timestamp'])
    
    # Step 2: Get intraday data for the current day if needed
    if not today_in_daily:
        intraday_df = fetch_alpha_vantage_intraday(symbol, interval='15min')
        
        if not intraday_df.empty:
            # Get the earliest intraday data point for today's open
            earliest_intraday = intraday_df.iloc[0]
            latest_intraday = intraday_df.iloc[-1]
            
            # Construct a daily bar for today based on intraday data
            today_ohlc = {
                'timestamp': pd.Timestamp(today),
                'open': earliest_intraday['open'],
                'high': intraday_df['high'].max(),
                'low': intraday_df['low'].min(),
                'close': latest_intraday['close'],
                'volume': intraday_df['volume'].sum(),
                'data_source': 'intraday_aggregated'
            }
            
            # Create a DataFrame for today's aggregated data
            today_df = pd.DataFrame([today_ohlc])
            
            # Append to historical daily data
            combined_df = pd.concat([daily_df, today_df], ignore_index=True)
        else:
            # No intraday data available, use historical only
            combined_df = daily_df.copy()
    else:
        # We already have today's data in the daily dataset
        combined_df = daily_df.copy()
    
    # Step 3: Update the latest price with real-time quote from Finnhub if available
    quote = get_current_quote(symbol)
    if quote and 'c' in quote and quote['c'] > 0:
        real_time_price = quote['c']
        
        # Update the most recent close price with real-time data
        last_idx = combined_df['timestamp'] == combined_df['timestamp'].max()
        combined_df.loc[last_idx, 'close'] = real_time_price
        combined_df.loc[last_idx, 'data_source'] = 'realtime'
        
        # If real-time price is higher/lower than the day's high/low, update those too
        if real_time_price > combined_df.loc[last_idx, 'high'].values[0]:
            combined_df.loc[last_idx, 'high'] = real_time_price
        
        if real_time_price < combined_df.loc[last_idx, 'low'].values[0]:
            combined_df.loc[last_idx, 'low'] = real_time_price
    
    return combined_df

# ENHANCEMENT 3: Market Regime Detection (continued)
def detect_market_regime(df):
    """
    Detect the current market regime (trending, ranging, or volatile)
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data and indicators
    
    Returns:
    str: The detected market regime - 'trending', 'ranging', or 'volatile'
    dict: Additional metrics about the regime detection
    """
    if df.empty or len(df) < 20:
        return "unknown", {}
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure ADX is calculated
    if 'adx' not in df.columns:
        # If we don't have ADX, calculate it
        # For simplicity, we'll use the existing calculate_indicators function
        df = calculate_indicators(df)
    
    # Get the last 20 bars for regime detection
    recent_df = df.iloc[-20:]
    
    # 1. Check ADX for trend strength
    adx_value = recent_df['adx'].iloc[-1]
    
    # 2. Check Bollinger Band width for volatility/ranging
    if 'bb_width' not in recent_df.columns:
        # Calculate if not available
        bb_width_mean = 0.05  # Default value
    else:
        bb_width = recent_df['bb_width'].iloc[-1]
        bb_width_mean = recent_df['bb_width'].mean()
    
    # 3. Check price movement relative to EMAs
    if 'ema50' in recent_df.columns and 'ema200' in recent_df.columns:
        # Calculate the angle/slope of EMA50
        ema50_slope = (recent_df['ema50'].iloc[-1] - recent_df['ema50'].iloc[-5]) / 5
        ema50_slope_pct = ema50_slope / recent_df['ema50'].iloc[-5] * 100
        
        # Check if price is consistently above/below EMAs (trend indicator)
        price_above_ema50 = recent_df['close'] > recent_df['ema50']
        price_above_ema200 = recent_df['close'] > recent_df['ema200']
        consistent_trend = (price_above_ema50.sum() > 15 or price_above_ema50.sum() < 5) and \
                           (price_above_ema200.sum() > 15 or price_above_ema200.sum() < 5)
    else:
        ema50_slope_pct = 0
        consistent_trend = False
    
    # 4. Check price volatility
    price_volatility = recent_df['high'].max() / recent_df['low'].min() - 1
    daily_ranges = (recent_df['high'] - recent_df['low']) / recent_df['low']
    avg_daily_range = daily_ranges.mean() * 100  # As percentage
    
    # Collect metrics for detailed analysis
    metrics = {
        'adx': adx_value,
        'bb_width': bb_width if 'bb_width' in locals() else 0,
        'bb_width_mean': bb_width_mean,
        'ema50_slope_pct': ema50_slope_pct,
        'consistent_trend': consistent_trend,
        'price_volatility': price_volatility,
        'avg_daily_range': avg_daily_range
    }
    
    # Decision logic for regime classification
    if adx_value > 25 and consistent_trend and abs(ema50_slope_pct) > 0.1:
        # Strong trend detected
        regime = 'trending'
    elif bb_width < 0.7 * bb_width_mean and adx_value < 20:
        # Tight consolidation/ranging
        regime = 'ranging'
    elif price_volatility > 0.08 or avg_daily_range > 3.0:
        # High volatility
        regime = 'volatile'
    elif adx_value < 15:
        # Low directional movement, likely ranging
        regime = 'ranging'
    else:
        # Default to weak trend if no strong signals
        regime = 'weak_trend'
    
    return regime, metrics

# Calculate all technical indicators
def calculate_indicators(df):
    if df.empty or len(df) < 20:  # Ensure we have enough data
        st.warning("Not enough data for reliable indicator calculation")
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate EMA values
    try:
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    except Exception as e:
        st.warning(f"Error calculating EMAs: {e}")
    
    # MACD calculation
    try:
        # Calculate MACD components
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add MACD crossover signals
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
    except Exception as e:
        st.warning(f"Error calculating MACD: {e}")
    
    # RSI calculation
    try:
        # Get price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over 14 periods
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, 0.001)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI SMA
        df['rsi_sma'] = df['rsi'].rolling(window=14).mean()
    except Exception as e:
        st.warning(f"Error calculating RSI: {e}")
    
    # Bollinger Bands
    try:
        # Calculate middle band (SMA20)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=20).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Calculate Bollinger Band width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate %B (where price is in relation to the bands)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    except Exception as e:
        st.warning(f"Error calculating Bollinger Bands: {e}")
    
    # EMA Cloud (bullish when ema8 > ema21)
    try:
        df['ema_cloud_bullish'] = df['ema8'] > df['ema21']
        df['ema_trend'] = df['ema50'] > df['ema200']
    except Exception as e:
        st.warning(f"Error calculating EMA Cloud: {e}")
    
    # Volume analysis
    try:
        # Volume moving average
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        
        # Volume Z-score
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_std'] = df['volume_std'].replace(0, 0.001)  # Avoid division by zero
        df['volume_z_score'] = (df['volume'] - df['volume_sma20']) / df['volume_std']
        
        # On Balance Volume (OBV)
        obv = pd.Series(index=df.index, dtype='float64')
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
    except Exception as e:
        st.warning(f"Error calculating volume metrics: {e}")
    
    # Stochastic RSI (Stoch RSI)
    try:
        # Calculate min and max RSI over the period
        min_rsi = df['rsi'].rolling(window=14).min()
        max_rsi = df['rsi'].rolling(window=14).max()
        
        # Handle division by zero
        rsi_range = max_rsi - min_rsi
        rsi_range = rsi_range.replace(0, 0.001)
        
        # Calculate Stochastic RSI
        df['stoch_rsi'] = (df['rsi'] - min_rsi) / rsi_range
        
        # Stoch RSI %K and %D
        df['stoch_rsi_k'] = df['stoch_rsi'] * 100
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    except Exception as e:
        st.warning(f"Error calculating Stochastic RSI: {e}")
    
    # ADX (Average Directional Index)
    try:
        # True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # +DM and -DM
        df['high_diff'] = df['high'] - df['high'].shift()
        df['low_diff'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = ((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0)) * df['high_diff']
        df['minus_dm'] = ((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0)) * df['low_diff']
        
        # Smoothed +DM, -DM and TR
        smooth_period = 14
        
        # Simple moving average of +DM, -DM and TR
        tr_sma = df['tr'].rolling(window=smooth_period).mean()
        tr_sma = tr_sma.replace(0, 0.001)  # Avoid division by zero
        
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=smooth_period).mean() / tr_sma
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=smooth_period).mean() / tr_sma
        
        # Calculate DX
        dx_denominator = df['plus_di'] + df['minus_di']
        dx_denominator = dx_denominator.replace(0, 0.001)  # Avoid division by zero
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / dx_denominator
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=smooth_period).mean()
    except Exception as e:
        st.warning(f"Error calculating ADX: {e}")
    
    return df

# ENHANCEMENT 1: Multi-timeframe analysis and alignment
def calculate_multi_timeframe_alignment(multi_tf_data):
    """
    Calculate alignment score across multiple timeframes
    
    Parameters:
    multi_tf_data (dict): Dictionary with data for multiple timeframes
    
    Returns:
    dict: Metrics about alignment across timeframes
    """
    if not multi_tf_data or len(multi_tf_data) < 2:
        return {
            'alignment_score': 0,
            'aligned': False,
            'timeframes_analyzed': 0
        }
    
    # Initialize alignment metrics
    alignment = {
        'ema_cloud': [],
        'macd': [],
        'rsi': [],
        'price_trend': [],
        'overall': []
    }
    
    # Process each timeframe
    for tf, df in multi_tf_data.items():
        if df.empty or len(df) < 20:
            continue
            
        # Calculate indicators if not already present
        if 'ema8' not in df.columns:
            df = calculate_indicators(df)
        
        # Get the last row for analysis
        last_row = df.iloc[-1]
        
        # EMA Cloud direction (ema8 > ema21 is bullish)
        if 'ema_cloud_bullish' in last_row:
            alignment['ema_cloud'].append(1 if last_row['ema_cloud_bullish'] else -1)
        
        # MACD direction
        if all(col in last_row for col in ['macd', 'macd_signal']):
            alignment['macd'].append(1 if last_row['macd'] > last_row['macd_signal'] else -1)
        
        # RSI direction
        if 'rsi' in last_row:
            # Neutral zone is 40-60, bullish above, bearish below
            rsi_value = last_row['rsi']
            if rsi_value > 60:
                alignment['rsi'].append(1)  # Bullish
            elif rsi_value < 40:
                alignment['rsi'].append(-1)  # Bearish
            else:
                alignment['rsi'].append(0)  # Neutral
        
        # Price trend (compared to ema50)
        if 'ema50' in last_row:
            alignment['price_trend'].append(1 if last_row['close'] > last_row['ema50'] else -1)
    
    # Calculate alignment scores
    for key in alignment:
        if key != 'overall' and alignment[key]:
            # Check if all values are the same (either all 1 or all -1)
            values = alignment[key]
            if all(v == values[0] for v in values) and values[0] != 0:
                # Perfect alignment (all bullish or all bearish)
                alignment[key] = {'aligned': True, 'direction': 'bullish' if values[0] == 1 else 'bearish', 'score': 1.0}
            elif sum(values) > 0 and all(v >= 0 for v in values):
                # Mostly bullish (some neutral but no bearish)
                alignment[key] = {'aligned': True, 'direction': 'bullish', 'score': sum(values) / len(values)}
            elif sum(values) < 0 and all(v <= 0 for v in values):
                # Mostly bearish (some neutral but no bullish)
                alignment[key] = {'aligned': True, 'direction': 'bearish', 'score': -sum(values) / len(values)}
            else:
                # Mixed signals
                alignment[key] = {'aligned': False, 'direction': 'mixed', 'score': 0}
    
    # Calculate overall alignment score
    aligned_keys = [k for k in alignment if k != 'overall' and isinstance(alignment[k], dict) and alignment[k]['aligned']]
    bullish_keys = [k for k in aligned_keys if alignment[k]['direction'] == 'bullish']
    bearish_keys = [k for k in aligned_keys if alignment[k]['direction'] == 'bearish']
    
    # Determine overall alignment
    if len(aligned_keys) >= 3:
        if len(bullish_keys) >= 3:
            overall_aligned = True
            overall_direction = 'bullish'
            overall_score = sum(alignment[k]['score'] for k in bullish_keys) / len(bullish_keys)
        elif len(bearish_keys) >= 3:
            overall_aligned = True
            overall_direction = 'bearish'
            overall_score = sum(alignment[k]['score'] for k in bearish_keys) / len(bearish_keys)
        else:
            overall_aligned = False
            overall_direction = 'mixed'
            overall_score = 0
    else:
        overall_aligned = False
        overall_direction = 'insufficient_data'
        overall_score = 0
    
    alignment['overall'] = {
        'aligned': overall_aligned,
        'direction': overall_direction,
        'score': overall_score,
        'timeframes_analyzed': len(multi_tf_data),
        'alignment_details': {k: alignment[k] for k in alignment if k != 'overall'}
    }
    
    return alignment['overall']

# ENHANCEMENT 2: Signal Clarity & Conflict Resolution
def calculate_signal_clarity(signals):
    """
    Calculate a signal clarity score based on agreement/disagreement among indicators
    
    Parameters:
    signals (list): List of signal dictionaries with indicator signals
    
    Returns:
    dict: Metrics about signal clarity and conflicts
    """
    if not signals:
        return {
            'clarity_score': 0,
            'agreement_ratio': 0,
            'conflicts': []
        }
    
    # Count buy and sell signals
    buy_signals = [s for s in signals if s['type'] == 'BUY']
    sell_signals = [s for s in signals if s['type'] == 'SELL']
    neutral_signals = [s for s in signals if s['type'] == 'NEUTRAL']
    
    total_directional = len(buy_signals) + len(sell_signals)
    total_signals = len(signals)
    
    # No directional signals
    if total_directional == 0:
        return {
            'clarity_score': 0,
            'agreement_ratio': 0,
            'conflicts': []
        }
    
    # Calculate agreement ratio (higher is better)
    if len(buy_signals) >= len(sell_signals):
        majority_direction = 'BUY'
        agreement_ratio = len(buy_signals) / total_directional
    else:
        majority_direction = 'SELL'
        agreement_ratio = len(sell_signals) / total_directional
    
    # Calculate clarity score (0-100)
    clarity_score = (agreement_ratio * 60) + (len(neutral_signals) / total_signals * 10)
    
    # Bonus for stronger agreement
    if agreement_ratio > 0.8:
        clarity_score += 20
    elif agreement_ratio > 0.6:
        clarity_score += 10
        
    # Cap at 100
    clarity_score = min(100, clarity_score)
    
    # Identify conflicting signals
    conflicts = []
    if agreement_ratio < 1.0:
        # There are some conflicting signals
        conflicting_direction = 'SELL' if majority_direction == 'BUY' else 'BUY'
        conflicting_signals = sell_signals if majority_direction == 'BUY' else buy_signals
        
        for signal in conflicting_signals:
            conflicts.append({
                'reason': signal['reason'],
                'points': signal['points'],
                'direction': signal['type']
            })
    
    return {
        'clarity_score': clarity_score,
        'agreement_ratio': agreement_ratio,
        'majority_direction': majority_direction,
        'conflicts': conflicts
    }

# ENHANCEMENT 4: Entry timing signals detection
def detect_entry_timing_signals(df, signal_direction):
    """
    Detect optimal entry timing signals on shorter timeframes
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicator data
    signal_direction (str): The overall signal direction ('BUY' or 'SELL')
    
    Returns:
    dict: Entry timing information
    """
    if df.empty or len(df) < 20:
        return {
            'entry_signal': False,
            'reason': 'Insufficient data'
        }
    
    # Get recent data (last 3 bars)
    recent = df.iloc[-3:]
    
    entry_signals = []
    
    # Look for entry signals based on overall direction
    if signal_direction == 'BUY':
        # For BUY signals
        
        # 1. MACD just crossed above signal line
        if 'macd_cross_up' in recent.columns and recent['macd_cross_up'].any():
            entry_signals.append({
                'type': 'bullish_macd_cross',
                'strength': 'strong'
            })
        
        # 2. RSI just bounced from oversold
        if 'rsi' in recent.columns:
            rsi_values = recent['rsi']
            if rsi_values.iloc[0] < 30 and rsi_values.iloc[-1] > 35:
                entry_signals.append({
                    'type': 'rsi_oversold_bounce',
                    'strength': 'strong'
                })
        
        # 3. Price just crossed above EMA8
        if 'ema8' in recent.columns:
            price_cross_above = (recent['close'].iloc[-1] > recent['ema8'].iloc[-1]) and \
                                (recent['close'].iloc[-2] <= recent['ema8'].iloc[-2])
            if price_cross_above:
                entry_signals.append({
                    'type': 'price_cross_above_ema8',
                    'strength': 'moderate'
                })
        
        # 4. Bollinger Band bounce from lower band
        if 'bb_lower' in recent.columns:
            bb_bounce = (recent['close'].iloc[-2] < recent['bb_lower'].iloc[-2]) and \
                        (recent['close'].iloc[-1] > recent['bb_lower'].iloc[-1])
            if bb_bounce:
                entry_signals.append({
                    'type': 'bollinger_band_bounce',
                    'strength': 'strong'
                })
        
    elif signal_direction == 'SELL':
        # For SELL signals
        
        # 1. MACD just crossed below signal line
        if 'macd_cross_down' in recent.columns and recent['macd_cross_down'].any():
            entry_signals.append({
                'type': 'bearish_macd_cross',
                'strength': 'strong'
            })
        
        # 2. RSI just peaked from overbought
        if 'rsi' in recent.columns:
            rsi_values = recent['rsi']
            if rsi_values.iloc[0] > 70 and rsi_values.iloc[-1] < 65:
                entry_signals.append({
                    'type': 'rsi_overbought_drop',
                    'strength': 'strong'
                })
        
        # 3. Price just crossed below EMA8
        if 'ema8' in recent.columns:
            price_cross_below = (recent['close'].iloc[-1] < recent['ema8'].iloc[-1]) and \
                                (recent['close'].iloc[-2] >= recent['ema8'].iloc[-2])
            if price_cross_below:
                entry_signals.append({
                    'type': 'price_cross_below_ema8',
                    'strength': 'moderate'
                })
        
        # 4. Bollinger Band breakdown from upper band
        if 'bb_upper' in recent.columns:
            bb_breakdown = (recent['close'].iloc[-2] > recent['bb_upper'].iloc[-2]) and \
                          (recent['close'].iloc[-1] < recent['bb_upper'].iloc[-1])
            if bb_breakdown:
                entry_signals.append({
                    'type': 'bollinger_band_breakdown',
                    'strength': 'strong'
                })
    
    # Return entry signal data
    if entry_signals:
        # Strong signals take precedence
        strong_signals = [s for s in entry_signals if s['strength'] == 'strong']
        if strong_signals:
            best_signal = strong_signals[0]
        else:
            best_signal = entry_signals[0]
            
        return {
            'entry_signal': True,
            'signal_type': best_signal['type'],
            'strength': best_signal['strength'],
            'all_signals': entry_signals
        }
    else:
        return {
            'entry_signal': False,
            'reason': 'No immediate entry signals detected'
        }

# Generate trading signals (continued)
def generate_signals(df, preferred_timeframe="short", multi_timeframe_data=None):
    """
    Enhanced signal generation with multi-timeframe analysis, regime detection,
    signal clarity and conflict resolution
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicator data
    preferred_timeframe (str): Preferred trading timeframe
    multi_timeframe_data (dict): Optional dictionary with data for multiple timeframes
    
    Returns:
    dict: Signal information with enhanced contextual awareness
    """
    if df.empty or len(df) < 20:
        return None
    
    signals = []
    confidence_score = 50  # Base confidence score
    
    # ENHANCEMENT 3: Detect market regime
    market_regime, regime_metrics = detect_market_regime(df)
    
    # Adjust indicator weights based on market regime and timeframe preference
    weights = {
        "trending": {
            "macd_crossover": 15,
            "ema_cloud": 15,
            "ema_trend": 20,
            "rsi": 10,
            "stoch_rsi": 5,
            "bb": 5,
            "volume": 15,
            "adx": 15
        },
        "ranging": {
            "macd_crossover": 5,
            "ema_cloud": 5,
            "ema_trend": 5,
            "rsi": 25,
            "stoch_rsi": 20,
            "bb": 25,
            "volume": 10,
            "adx": 5
        },
        "volatile": {
            "macd_crossover": 10,
            "ema_cloud": 10,
            "ema_trend": 10,
            "rsi": 15,
            "stoch_rsi": 15,
            "bb": 15,
            "volume": 20,
            "adx": 5
        },
        "weak_trend": {
            "macd_crossover": 15,
            "ema_cloud": 15,
            "ema_trend": 10,
            "rsi": 15,
            "stoch_rsi": 15,
            "bb": 10,
            "volume": 15,
            "adx": 5
        },
        "unknown": {
            "macd_crossover": 15,
            "ema_cloud": 10,
            "ema_trend": 10,
            "rsi": 15,
            "stoch_rsi": 10,
            "bb": 10,
            "volume": 15,
            "adx": 15
        }
    }
    
    # Use weights based on detected market regime
    w = weights[market_regime]
    
    # Check last row for signals
    last_idx = -1
    
    try:
        # MACD Signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            # Bullish MACD crossover
            if df['macd_cross_up'].iloc[last_idx]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'MACD Bullish Crossover',
                    'points': w["macd_crossover"],
                    'timeframe': TIMEFRAMES["medium"]
                })
                confidence_score += w["macd_crossover"]
            
            # Bearish MACD crossover
            elif df['macd_cross_down'].iloc[last_idx]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'MACD Bearish Crossover',
                    'points': w["macd_crossover"],
                    'timeframe': TIMEFRAMES["medium"]
                })
                confidence_score += w["macd_crossover"]
        
        # EMA Cloud
        if 'ema_cloud_bullish' in df.columns:
            if df['ema_cloud_bullish'].iloc[last_idx]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'EMA Cloud Bullish (EMA8 > EMA21)',
                    'points': w["ema_cloud"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["ema_cloud"]
            else:
                signals.append({
                    'type': 'SELL',
                    'reason': 'EMA Cloud Bearish (EMA8 < EMA21)',
                    'points': w["ema_cloud"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["ema_cloud"]
        
        # EMA50 vs EMA200 (Golden/Death Cross)
        if 'ema_trend' in df.columns:
            if df['ema_trend'].iloc[last_idx]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Long-term Bullish Trend (EMA50 > EMA200)',
                    'points': w["ema_trend"],
                    'timeframe': TIMEFRAMES["long"]
                })
                confidence_score += w["ema_trend"]
            else:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Long-term Bearish Trend (EMA50 < EMA200)',
                    'points': w["ema_trend"],
                    'timeframe': TIMEFRAMES["long"]
                })
                confidence_score += w["ema_trend"]
        
        # RSI Signals
        if 'rsi' in df.columns:
            # Oversold
            if df['rsi'].iloc[last_idx] < 30:
                signals.append({
                    'type': 'BUY',
                    'reason': 'RSI Oversold (<30)',
                    'points': w["rsi"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["rsi"]
            # Overbought
            elif df['rsi'].iloc[last_idx] > 70:
                signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought (>70)',
                    'points': w["rsi"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["rsi"]
        
        # Stochastic RSI Signals
        if 'stoch_rsi_k' in df.columns and 'stoch_rsi_d' in df.columns:
            # Oversold
            if df['stoch_rsi_k'].iloc[last_idx] < 20:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Stochastic RSI Oversold (<20)',
                    'points': w["stoch_rsi"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["stoch_rsi"]
            # Overbought
            elif df['stoch_rsi_k'].iloc[last_idx] > 80:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Stochastic RSI Overbought (>80)',
                    'points': w["stoch_rsi"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["stoch_rsi"]
            
            # Crossover
            if df['stoch_rsi_k'].iloc[last_idx] > df['stoch_rsi_d'].iloc[last_idx] and df['stoch_rsi_k'].iloc[last_idx-1] <= df['stoch_rsi_d'].iloc[last_idx-1]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Stochastic RSI Bullish Crossover',
                    'points': w["stoch_rsi"] - 5,  # Slightly less weight for crossover
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["stoch_rsi"] - 5
            elif df['stoch_rsi_k'].iloc[last_idx] < df['stoch_rsi_d'].iloc[last_idx] and df['stoch_rsi_k'].iloc[last_idx-1] >= df['stoch_rsi_d'].iloc[last_idx-1]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Stochastic RSI Bearish Crossover',
                    'points': w["stoch_rsi"] - 5,  # Slightly less weight for crossover
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["stoch_rsi"] - 5
        
        # Bollinger Band signals
        if 'bb_pct_b' in df.columns and 'bb_width' in df.columns:
            # Price near upper band
            if df['bb_pct_b'].iloc[last_idx] > 0.9:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Price near upper Bollinger Band',
                    'points': w["bb"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["bb"]
            # Price near lower band
            elif df['bb_pct_b'].iloc[last_idx] < 0.1:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Price near lower Bollinger Band',
                    'points': w["bb"],
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["bb"]
            
            # Bollinger Band squeeze (setup for volatility breakout)
            if df['bb_width'].iloc[last_idx] < df['bb_width'].rolling(window=20).mean().iloc[last_idx] * 0.8:
                signals.append({
                    'type': 'NEUTRAL',
                    'reason': 'Bollinger Band Squeeze (potential breakout setup)',
                    'points': w["bb"] - 5,  # Slightly less weight for squeeze
                    'timeframe': TIMEFRAMES["short"]
                })
                confidence_score += w["bb"] - 5
        
        # Volume confirmation
        if 'volume_z_score' in df.columns:
            if df['volume_z_score'].iloc[last_idx] > 1.5:
                # High volume confirms the direction
                direction = 'BUY' if df['close'].iloc[last_idx] > df['close'].iloc[last_idx-1] else 'SELL'
                signals.append({
                    'type': direction,
                    'reason': 'High Volume Confirmation',
                    'points': w["volume"],
                    'timeframe': "Enhances existing signals"
                })
                confidence_score += w["volume"]
        
        # ADX - Strong trend
        if 'adx' in df.columns:
            if df['adx'].iloc[last_idx] > 25:
                signals.append({
                    'type': 'NEUTRAL',
                    'reason': 'Strong Trend (ADX > 25)',
                    'points': w["adx"],
                    'timeframe': "Confirms directional signals"
                })
                confidence_score += w["adx"]
        
        # ENHANCEMENT 2: Calculate signal clarity score
        signal_clarity = calculate_signal_clarity(signals)
        
        # ENHANCEMENT 1: Apply multi-timeframe alignment boost if available
        multi_timeframe_alignment = None
        if multi_timeframe_data and len(multi_timeframe_data) >= 2:
            multi_timeframe_alignment = calculate_multi_timeframe_alignment(multi_timeframe_data)
            
            # Boost confidence if timeframes align
            if multi_timeframe_alignment['aligned']:
                alignment_boost = multi_timeframe_alignment['score'] * 10  # 0-10 point boost
                confidence_score += alignment_boost
                
                # Add a reason for the alignment
                signals.append({
                    'type': 'BUY' if multi_timeframe_alignment['direction'] == 'bullish' else 'SELL',
                    'reason': f"Multi-timeframe alignment ({multi_timeframe_alignment['timeframes_analyzed']} timeframes)",
                    'points': alignment_boost,
                    'timeframe': "All analyzed timeframes"
                })
        
        # Apply signal clarity adjustment to confidence
        if signal_clarity['clarity_score'] > 75:
            # High clarity - boost confidence
            clarity_boost = (signal_clarity['clarity_score'] - 75) / 5  # Up to 5 points boost
            confidence_score += clarity_boost
        elif signal_clarity['clarity_score'] < 40:
            # Low clarity - reduce confidence
            clarity_penalty = (40 - signal_clarity['clarity_score']) / 4  # Up to 10 points penalty
            confidence_score -= clarity_penalty
            
        # ENHANCEMENT 5: Apply historical performance adjustment if available
        signal_key = None
        historical_performance = None
        
        # Determine overall signal direction for historical lookup
        buy_points = sum(signal['points'] for signal in signals if signal['type'] == 'BUY')
        sell_points = sum(signal['points'] for signal in signals if signal['type'] == 'SELL')
        
        if buy_points > sell_points:
            preliminary_direction = 'BUY'
        elif sell_points > buy_points:
            preliminary_direction = 'SELL'
        else:
            preliminary_direction = 'NEUTRAL'
            
        # Create a temporary signal for historical lookup
        temp_signal = {
            'direction': preliminary_direction,
            'reasons': [s['reason'] for s in signals if s['type'] == preliminary_direction or s['type'] == 'NEUTRAL'],
            'market_regime': market_regime
        }
        
        signal_key = generate_signal_key(temp_signal)
        if signal_key:
            historical_performance = get_signal_historical_performance(temp_signal)
            
            if historical_performance and historical_performance['total'] >= 5:
                # We have enough history to make an adjustment
                win_rate = historical_performance['win_rate']
                
                if win_rate > 60:
                    # Above average win rate - boost confidence
                    perf_boost = (win_rate - 60) / 4  # Up to 10 points boost for excellent win rates
                    confidence_score += perf_boost
                elif win_rate < 40:
                    # Below average win rate - reduce confidence
                    perf_penalty = (40 - win_rate) / 4  # Up to 10 points penalty
                    confidence_score -= perf_penalty
        
        # Calculate final confidence (cap at 95)
        final_confidence = min(95, max(5, confidence_score))
        
        # Determine overall timeframe recommendation based on preferred timeframe
        recommended_timeframe = TIMEFRAMES[preferred_timeframe]
        
        # Determine overall signal
        if buy_points > sell_points and final_confidence >= 60:
            # ENHANCEMENT 4: Check for entry timing signals
            entry_signal = detect_entry_timing_signals(df, 'BUY')
            
            overall_signal = {
                'direction': 'BUY',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'BUY' or s['type'] == 'NEUTRAL'],
                'timeframe': recommended_timeframe,
                'market_regime': market_regime,
                'regime_metrics': regime_metrics,
                'signal_clarity': signal_clarity,
                'entry_signals': entry_signal,
                'multi_timeframe_alignment': multi_timeframe_alignment,
                'historical_performance': historical_performance
            }
        elif sell_points > buy_points and final_confidence >= 60:
            # ENHANCEMENT 4: Check for entry timing signals
            entry_signal = detect_entry_timing_signals(df, 'SELL')
            
            overall_signal = {
                'direction': 'SELL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'SELL' or s['type'] == 'NEUTRAL'],
                'timeframe': recommended_timeframe,
                'market_regime': market_regime,
                'regime_metrics': regime_metrics,
                'signal_clarity': signal_clarity,
                'entry_signals': entry_signal,
                'multi_timeframe_alignment': multi_timeframe_alignment,
                'historical_performance': historical_performance
            }
        else:
            overall_signal = {
                'direction': 'NEUTRAL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'NEUTRAL'],
                'timeframe': "Wait for clearer signals",
                'market_regime': market_regime,
                'regime_metrics': regime_metrics,
                'signal_clarity': signal_clarity,
                'multi_timeframe_alignment': multi_timeframe_alignment,
                'historical_performance': historical_performance
            }
        
        return overall_signal
    
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return None

# Enhanced signal generation with comprehensive data
def generate_enhanced_signals(stocks, min_confidence=65, timeframe="short"):
    all_signals = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(stocks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
            
            # Add delay to avoid API rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(12)
            
            # ENHANCEMENT 1: Fetch multi-timeframe data
            multi_timeframe_data = fetch_multi_timeframe_data(symbol, timeframe)
            
            # Get comprehensive data for the primary timeframe
            df = create_comprehensive_dataset(symbol, timeframe)
            if df.empty:
                continue
            
            # Store the price information
            last_timestamp = df['timestamp'].max()
            last_row = df[df['timestamp'] == last_timestamp].iloc[0]
            current_price = last_row['close']
            data_source = last_row['data_source']
            
            # Get the previous day's close for comparison
            prev_day_rows = df[(df['timestamp'] < last_timestamp) & (df['data_source'] == 'daily')]
            if not prev_day_rows.empty:
                prev_close = prev_day_rows.iloc[-1]['close']
                prev_date = prev_day_rows.iloc[-1]['timestamp']
            else:
                prev_close = current_price
                prev_date = last_timestamp
            
            # Calculate indicators with the updated data
            df = calculate_indicators(df)
            
            # Generate signal with enhanced features
            signal = generate_signals(df, timeframe, multi_timeframe_data)
            
            if signal and signal['confidence'] >= min_confidence and signal['direction'] != 'NEUTRAL':
                # Add price and data source information to the signal data
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                signal['last_price'] = current_price
                signal['prev_close'] = prev_close
                signal['prev_date'] = prev_date.strftime("%Y-%m-%d")
                signal['price_diff_pct'] = ((current_price - prev_close) / prev_close) * 100
                signal['data_source'] = data_source
                
                # Adjust confidence if prices have significant difference
                price_diff_abs = abs(signal['price_diff_pct'])
                
                if price_diff_abs > 2.0:  # More than 2% difference
                    # Flag for significant price movement since last close
                    signal['price_warning'] = True
                    
                    # For large moves that align with the signal direction
                    if (signal['direction'] == 'BUY' and current_price > prev_close) or \
                       (signal['direction'] == 'SELL' and current_price < prev_close):
                        # Price moved in favor of signal - reinforce confidence
                        signal['confidence'] = min(95, signal['confidence'] + 5)
                        signal['confidence_adjustment'] = "Increased"
                    else:
                        # Price moved against signal - reduce confidence
                        signal['confidence'] = max(50, signal['confidence'] - int(price_diff_abs))
                        signal['confidence_adjustment'] = "Decreased"
                
                all_signals.append(signal)
        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            continue
    
    # Sort by confidence score
    all_signals.sort(key=lambda x: x['confidence'], reverse=True)
    return all_signals

# Function to scan for signals (using comprehensive data)
def scan_for_signals(stocks, min_confidence=65, timeframe="short"):
    all_signals = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(stocks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
            
            # Add delay to avoid API rate limits
            if i > 0 and i % 5 == 0:
                time.sleep(12)
            
            # ENHANCEMENT 1: Fetch multi-timeframe data for enhanced accuracy
            multi_timeframe_data = fetch_multi_timeframe_data(symbol, timeframe)
            
            # Get comprehensive data
            df = create_comprehensive_dataset(symbol, timeframe)
            if df.empty:
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Generate enhanced signal
            signal = generate_signals(df, timeframe, multi_timeframe_data)
            
            if signal and signal['confidence'] >= min_confidence and signal['direction'] != 'NEUTRAL':
                # Add symbol and price info
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Get the current price from the last row
                last_timestamp = df['timestamp'].max()
                signal['last_price'] = df[df['timestamp'] == last_timestamp]['close'].values[0]
                signal['data_source'] = df[df['timestamp'] == last_timestamp]['data_source'].values[0]
                
                all_signals.append(signal)
        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            continue
    
    # Sort by confidence score
    all_signals.sort(key=lambda x: x['confidence'], reverse=True)
    return all_signals

# Paper trading functions
def execute_paper_trade(symbol, direction, price, confidence, timeframe):
    portfolio = st.session_state.paper_portfolio
    positions = portfolio['positions']
    
    # Calculate trade size based on confidence (higher confidence = larger position)
    # Use 2% of portfolio for 70% confidence, scale up to 5% for 95% confidence
    position_size_pct = 0.02 + (confidence - 70) / 25 * 0.03 if confidence >= 70 else 0.02
    position_size_pct = min(position_size_pct, 0.05)  # Cap at 5%
    
    trade_value = portfolio['cash'] * position_size_pct
    shares = int(trade_value / price)
    
    # If shares is 0, use at least 1 share
    shares = max(shares, 1)
    
    # Execute the trade
    if direction == 'BUY':
        # Check if we already have a position in this stock
        if symbol in positions:
            # If we have a SELL position, close it (fully or partially)
            if positions[symbol]['direction'] == 'SELL':
                # Close position
                close_paper_trade(symbol, price, "Signal Reversal")
            else:
                # Add to existing position
                current_shares = positions[symbol]['shares']
                current_value = current_shares * positions[symbol]['entry_price']
                new_value = shares * price
                total_value = current_value + new_value
                total_shares = current_shares + shares
                # Calculate new average entry price
                positions[symbol]['entry_price'] = total_value / total_shares
                positions[symbol]['shares'] = total_shares
                portfolio['cash'] -= shares * price
        else:
            # Open new position
            positions[symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'direction': 'BUY',
                'signal_confidence': confidence,
                'timeframe': timeframe,
                'signal_key': ''  # Will be filled by tracking function
            }
            portfolio['cash'] -= shares * price
    
    elif direction == 'SELL':
        # For simplicity, let's assume we can short stocks
        if symbol in positions:
            # If we have a BUY position, close it
            if positions[symbol]['direction'] == 'BUY':
                # Close position
                close_paper_trade(symbol, price, "Signal Reversal")
            else:
                # Add to existing short position
                current_shares = positions[symbol]['shares']
                current_value = current_shares * positions[symbol]['entry_price']
                new_value = shares * price
                total_value = current_value + new_value
                total_shares = current_shares + shares
                # Calculate new average entry price
                positions[symbol]['entry_price'] = total_value / total_shares
                positions[symbol]['shares'] = total_shares
                portfolio['cash'] += shares * price  # Add cash for short selling
        else:
            # Open new short position
            positions[symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'direction': 'SELL',
                'signal_confidence': confidence,
                'timeframe': timeframe,
                'signal_key': ''  # Will be filled by tracking function
            }
            portfolio['cash'] += shares * price  # Add cash for short selling
    
    # Save to disk after trade
    save_paper_trading_data()
    
    return shares

# Function to close paper trades
def close_paper_trade(symbol, current_price, reason="Manual"):
    portfolio = st.session_state.paper_portfolio
    positions = portfolio['positions']
    
    if symbol in positions:
        position = positions[symbol]
        shares = position['shares']
        
        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl = (current_price - position['entry_price']) * shares
        else:  # SELL/SHORT
            pnl = (position['entry_price'] - current_price) * shares
        
        pnl_pct = (pnl / (position['entry_price'] * shares)) * 100
        
        # Update cash
        if position['direction'] == 'BUY':
            portfolio['cash'] += shares * current_price
        else:  # SELL/SHORT
            portfolio['cash'] -= shares * current_price
        
        # Record trade in history
        trade_record = {
            'symbol': symbol,
            'direction': position['direction'],
            'shares': shares,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'confidence': position['signal_confidence'],
            'timeframe': position.get('timeframe', 'Unknown'),
            'signal_key': position.get('signal_key', '')
        }
        
        portfolio['trade_history'].append(trade_record)
        
        # Update performance metrics
        portfolio['performance']['trades'] += 1
        portfolio['performance']['total_return'] += pnl
        
        # Calculate win rate
        winning_trades = sum(1 for trade in portfolio['trade_history'] if trade['pnl'] > 0)
        total_trades = len(portfolio['trade_history'])
        portfolio['performance']['win_rate'] = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # ENHANCEMENT 5: Update signal performance tracking if we have a signal key
        if 'signal_key' in position and position['signal_key']:
            # Update historical performance of this signal type
            update_signal_performance(position['signal_key'], pnl > 0)
        
        # Remove position
        del positions[symbol]
        
        # Save to disk after closing trade
        save_paper_trading_data()
        
        return pnl, pnl_pct
    
    return 0, 0

# Function to update paper trading positions with real-time prices
def update_paper_positions(quotes_data=None):
    """Update all positions with latest price data"""
    if not quotes_data:
        # If no quotes_data provided, try to get current quotes for all positions
        quotes_data = {}
        for symbol in st.session_state.paper_portfolio['positions']:
            quote = get_current_quote(symbol)
            if quote and 'c' in quote:
                quotes_data[symbol] = quote['c']
    else:
        # Convert quotes_data from DataFrame to dict for easier lookups
        if isinstance(quotes_data, pd.DataFrame):
            quotes_dict = {}
            for _, row in quotes_data.iterrows():
                quotes_dict[row['Symbol']] = row['Price']
            quotes_data = quotes_dict
    
    portfolio = st.session_state.paper_portfolio
    positions = portfolio['positions']
    
    # Calculate current values and P&L
    current_portfolio_value = portfolio['cash']
    for symbol, position in positions.items():
        if symbol in quotes_data:
            current_price = quotes_data[symbol]
            # Calculate P&L for this position
            if position['direction'] == 'BUY':
                position['current_price'] = current_price
                position['pnl'] = (current_price - position['entry_price']) * position['shares']
                position['pnl_pct'] = ((current_price / position['entry_price']) - 1) * 100
                current_portfolio_value += position['shares'] * current_price
            else:  # SELL/SHORT
                position['current_price'] = current_price
                position['pnl'] = (position['entry_price'] - current_price) * position['shares']
                position['pnl_pct'] = ((position['entry_price'] / current_price) - 1) * 100
                current_portfolio_value -= position['shares'] * current_price
    
    # Calculate overall portfolio P&L
    portfolio['current_value'] = current_portfolio_value
    portfolio['pnl'] = current_portfolio_value - 100000  # Compared to starting value
    portfolio['pnl_pct'] = (portfolio['pnl'] / 100000) * 100
    
    return portfolio

# Fixed version of color_pnl function for dataframe styling (continued)
def color_pnl(val):
    try:
        val_str = str(val)
        if 'P&L' in val_str or 'P&L %' in val_str or '$' in val_str or '%' in val_str:
            if '%' in val_str:
                val_num = float(val_str.strip('%'))
            elif '$' in val_str:
                val_num = float(val_str.strip('$'))
            else:
                val_num = float(val)
            
            color = 'green' if val_num > 0 else 'red' if val_num < 0 else 'black'
            return f'color: {color}'
    except:
        return ''
    return ''

# Plot chart with indicators
def plot_chart(df, symbol):
    if df.empty or len(df) < 20:
        st.error("Not enough data to plot chart")
        return None
    
    try:
        # Create subplots
        fig = make_subplots(rows=4, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.02, 
                           row_heights=[0.5, 0.15, 0.15, 0.2])
        
        # Add price candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if available
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(250, 0, 0, 0.5)'),
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_middle'],
                    name='BB Middle',
                    line=dict(color='rgba(0, 0, 250, 0.5)'),
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(0, 250, 0, 0.5)'),
                ),
                row=1, col=1
            )
        
        # Add EMAs if available
        if 'ema8' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['ema8'],
                    name='EMA 8',
                    line=dict(color='purple', width=1),
                ),
                row=1, col=1
            )
        
        if 'ema21' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['ema21'],
                    name='EMA 21',
                    line=dict(color='orange', width=1),
                ),
                row=1, col=1
            )
        
        # Add MACD
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd'],
                    name='MACD',
                    line=dict(color='blue', width=1.5),
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd_signal'],
                    name='MACD Signal',
                    line=dict(color='red', width=1.5),
                ),
                row=2, col=1
            )
            
            # MACD histogram
            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['macd_hist'],
                    name='MACD Histogram',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # Add RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rsi'],
                    name='RSI',
                    line=dict(color='blue', width=1),
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold levels
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=[70] * len(df),
                    name='Overbought',
                    line=dict(color='red', width=1, dash='dash'),
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=[30] * len(df),
                    name='Oversold',
                    line=dict(color='green', width=1, dash='dash'),
                ),
                row=3, col=1
            )
        
        # Add Volume
        colors = ['green' if df['close'].iloc[i] >= df['close'].iloc[i-1] else 'red' 
                 for i in range(1, len(df))]
        colors.insert(0, 'green')  # Add color for the first bar
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=4, col=1
        )
        
        # Add OBV if available
        if 'obv' in df.columns:
            # Scale OBV to fit with volume for visualization
            if df['obv'].abs().max() > 0:  # Avoid division by zero
                obv_scaled = df['obv'] / df['obv'].abs().max() * df['volume'].max() * 0.7
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=obv_scaled,
                        name='OBV (scaled)',
                        line=dict(color='purple', width=1.5),
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Chart with Comprehensive Data',
            xaxis_title='Date',
            yaxis_title='Price',
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# Real-time Stock Data Screen
def setup_real_time_monitoring(symbols):
    # Initialize session state to track if websocket is running
    if 'ws_running' not in st.session_state:
        st.session_state.ws_running = False
        st.session_state.ws_symbols = symbols
    
    # Start websocket if not running
    if not st.session_state.ws_running:
        st.session_state.ws = init_websocket()
        st.session_state.ws_running = True
    
    # Display real-time data
    st.subheader("Real-time Trade Data")
    
    # Create container for real-time data
    real_time_container = st.container()
    
    # Process and display new trades
    with real_time_container:
        # Check for new messages in the queue
        trades = []
        while not ws_message_queue.empty():
            try:
                trade = ws_message_queue.get_nowait()
                trades.append(trade)
                if len(trades) >= 10:  # Limit to 10 trades to avoid UI lag
                    break
            except queue.Empty:
                break
        
        # Display trades
        if trades:
            df_trades = pd.DataFrame(trades)
            if not df_trades.empty:
                st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("Waiting for real-time trade data...")

# Paper Trading Page
def show_paper_trading():
    st.title("📝 Paper Trading Portfolio")
    
    # Initialize paper trading
    initialize_paper_trading()
    
    # Sidebar options
    st.sidebar.header("Paper Trading Settings")
    
    # Add save portfolio button
    if st.sidebar.button("Save Portfolio Data"):
        save_paper_trading_data()
    
    # Add reset portfolio button
    if st.sidebar.button("Reset Portfolio"):
        st.session_state.paper_portfolio = {
            'cash': 100000,
            'positions': {},
            'trade_history': [],
            'performance': {'total_return': 0, 'win_rate': 0, 'trades': 0}
        }
        save_paper_trading_data()
        st.sidebar.success("Portfolio reset to $100,000")
    
    # Option to execute trades from last scanner results
    if 'last_signals' in st.session_state and st.session_state.last_signals:
        signals_count = len(st.session_state.last_signals)
        if st.sidebar.button(f"Execute Last {signals_count} Scanner Signals"):
            for signal in st.session_state.last_signals:
                symbol = signal['symbol']
                direction = signal['direction']
                price = signal['last_price']
                confidence = signal['confidence']
                timeframe = signal.get('timeframe', 'Medium-term')
                
                # Execute the paper trade
                shares = execute_paper_trade(symbol, direction, price, confidence, timeframe)
                
                # ENHANCEMENT 5: Store signal key for performance tracking
                signal_key = generate_signal_key(signal)
                if signal_key and symbol in st.session_state.paper_portfolio['positions']:
                    st.session_state.paper_portfolio['positions'][symbol]['signal_key'] = signal_key
                
                st.sidebar.info(f"Executed: {direction} {shares} shares of {symbol} at ${price:.2f}")
    
    # Main area - Summary
    col1, col2, col3 = st.columns(3)
    
    # Get latest prices and update positions
    portfolio = update_paper_positions()
    
    with col1:
        st.metric("Cash Balance", f"${portfolio['cash']:.2f}")
    
    with col2:
        st.metric("Portfolio Value", f"${portfolio.get('current_value', 0):.2f}")
    
    with col3:
        pnl = portfolio.get('pnl', 0)
        pnl_pct = portfolio.get('pnl_pct', 0)
        st.metric("Total P&L", f"${pnl:.2f} ({pnl_pct:.2f}%)", delta=f"{pnl_pct:.2f}%")
    
    # Open positions
    st.subheader("Open Positions")
    
    if not portfolio['positions']:
        st.info("No open positions. Execute trades from the Scanner page or add positions manually.")
    else:
        positions_data = []
        for symbol, position in portfolio['positions'].items():
            positions_data.append({
                'Symbol': symbol,
                'Direction': position['direction'],
                'Shares': position['shares'],
                'Entry Price': f"${position['entry_price']:.2f}",
                'Current Price': f"${position.get('current_price', position['entry_price']):.2f}",
                'Entry Date': position['entry_time'],
                'P&L': f"${position.get('pnl', 0):.2f}",
                'P&L %': f"{position.get('pnl_pct', 0):.2f}%",
                'Confidence': f"{position.get('signal_confidence', 0)}%",
                'Timeframe': position.get('timeframe', 'Unknown')
            })
        
        df_positions = pd.DataFrame(positions_data)
        
        # Apply styling and display
        try:
            st.dataframe(df_positions.style.applymap(color_pnl), use_container_width=True)
        except Exception as e:
            st.error(f"Error styling dataframe: {e}")
            st.dataframe(df_positions, use_container_width=True)
        
        # Add close position buttons
        if len(portfolio['positions']) > 0:
            selected_position = st.selectbox("Select position to close:", list(portfolio['positions'].keys()))
            
            if st.button(f"Close {selected_position} Position"):
                position = portfolio['positions'][selected_position]
                current_price = position.get('current_price', position['entry_price'])
                pnl, pnl_pct = close_paper_trade(selected_position, current_price)
                
                st.success(f"Closed {selected_position} position with P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                st.rerun()  # Refresh the page
    
    # Trade history
    st.subheader("Trade History")
    
    if not portfolio['trade_history']:
        st.info("No closed trades yet.")
    else:
        history_data = []
        for trade in portfolio['trade_history']:
            history_data.append({
                'Symbol': trade['symbol'],
                'Direction': trade['direction'],
                'Shares': trade['shares'],
                'Entry Price': f"${trade['entry_price']:.2f}",
                'Exit Price': f"${trade['exit_price']:.2f}",
                'Entry Time': trade['entry_time'],
                'Exit Time': trade['exit_time'],
                'P&L': f"${trade['pnl']:.2f}",
                'P&L %': f"{trade['pnl_pct']:.2f}%",
                'Confidence': f"{trade.get('confidence', 0)}%",
                'Reason': trade['reason']
            })
        
        df_history = pd.DataFrame(history_data)
        
        # Apply styling and display
        try:
            st.dataframe(df_history.style.applymap(color_pnl), use_container_width=True)
        except Exception as e:
            st.error(f"Error styling history dataframe: {e}")
            st.dataframe(df_history, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Total Trades", portfolio['performance']['trades'])
    
    with metrics_col2:
        st.metric("Win Rate", f"{portfolio['performance']['win_rate']:.2f}%")
    
    with metrics_col3:
        avg_return = 0
        if portfolio['performance']['trades'] > 0:
            avg_return = portfolio['performance']['total_return'] / portfolio['performance']['trades']
        st.metric("Average Return Per Trade", f"${avg_return:.2f}")
    
    # ENHANCEMENT 5: Display Signal Performance Tracking
    if 'signal_performance' in st.session_state and st.session_state.signal_performance.get('signals'):
        st.subheader("Signal Performance Tracking")
        
        signal_perf = st.session_state.signal_performance['signals']
        
        if signal_perf:
            signal_data = []
            for key, data in signal_perf.items():
                if data['total'] >= 3:  # Only show signals with sufficient data
                    signal_data.append({
                        'Signal Type': key[:40] + "..." if len(key) > 40 else key,
                        'Win Rate': f"{data['win_rate']:.2f}%",
                        'Wins': data['wins'],
                        'Losses': data['losses'],
                        'Total Trades': data['total']
                    })
            
            if signal_data:
                df_signals = pd.DataFrame(signal_data)
                st.dataframe(df_signals, use_container_width=True)
            else:
                st.info("Not enough signal performance data accumulated yet.")
        else:
            st.info("No signal performance data available yet.")
    
    # Add auto-refresh option
    refresh_options = st.radio(
        "Auto-refresh portfolio", 
        ["None", "Every 5 seconds", "Every 25 minutes"], 
        index=0,
        horizontal=True
    )
    
    if refresh_options == "Every 5 seconds":
        time.sleep(5)  # Refresh every 5 seconds
        st.rerun()
    elif refresh_options == "Every 25 minutes":
        st.write(f"Next refresh in 25 minutes")
        time.sleep(1500)  # 25 minutes = 1500 seconds
        st.rerun()

# App navigation
def main():
    # First check API keys
    av_valid, fh_valid = verify_api_keys()
    
    if not av_valid:
        st.error("Alpha Vantage API key is invalid or has reached its limit. Historical data will not be available.")
    
    if not fh_valid:
        st.warning("Finnhub API key is invalid or has reached its limit. Real-time data may not be available.")
    
    # Initialize stock universe
    load_stock_universe()
    
    # Initialize paper trading
    initialize_paper_trading()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .buy-signal {
        color: green;
        font-weight: bold;
    }
    .sell-signal {
        color: red;
        font-weight: bold;
    }
    .neutral-signal {
        color: orange;
        font-weight: bold;
    }
    .timeframe-info {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .api-info {
        background-color: #e7f3fe;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #2196F3;
    }
    .price-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #ffc107;
    }
    .data-source-info {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #4CAF50;
    }
    .regime-info {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #9c27b0;
    }
    .signal-clarity-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #2962ff;
    }
    .multi-timeframe-info {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #ffb300;
    }
    .entry-signal-info {
        background-color: #fce4ec;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #e91e63;
    }
    .historical-info {
        background-color: #efebe9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #795548;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Scanner", "Paper Trading", "Real-time Monitor", "Documentation"])
    
    # Display note about APIs
    st.sidebar.markdown("""
    <div class="api-info">
    <strong>Data Sources:</strong><br>
    - Historical daily data: Alpha Vantage<br>
    - Intraday data: Alpha Vantage<br>
    - Real-time data: Finnhub<br>
    <small>Free tier: 25 API calls per day</small>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Scanner":
        show_scanner()
    elif page == "Paper Trading":
        show_paper_trading()
    elif page == "Real-time Monitor":
        show_real_time_monitor()
    elif page == "Documentation":
        show_documentation()

# Dashboard page
def show_dashboard():
    st.markdown('<p class="big-font">📈 Enhanced Technical Trading Signal Dashboard</p>', unsafe_allow_html=True)
    
    # Settings in sidebar
    st.sidebar.header("Dashboard Settings")
    
    # Trading timeframe selection
    timeframe = st.sidebar.radio(
        "Trading Timeframe",
        ["Short-term (1-7 days)", "Medium-term (1-4 weeks)", "Long-term (1-3 months)"],
        index=0  # Default to short-term
    )
    
    # Map display timeframe to internal timeframe code
    timeframe_map = {
        "Short-term (1-7 days)": "short",
        "Medium-term (1-4 weeks)": "medium",
        "Long-term (1-3 months)": "long"
    }
    selected_timeframe = timeframe_map[timeframe]
    
    # Stock selection from all available US stocks
    stock_symbols = [s[0] for s in st.session_state.stock_universe]
    stock_names = [f"{s[0]}: {s[1]}" for s in st.session_state.stock_universe]
    
    # Create a search box for stocks
    stock_search = st.sidebar.text_input("Search for a stock symbol or name:")
    
    if stock_search:
        # Filter stocks based on search
        filtered_stocks = [(s[0], s[1]) for s in st.session_state.stock_universe 
                          if stock_search.upper() in s[0] or stock_search.lower() in s[1].lower()]
        
        if not filtered_stocks:
            st.sidebar.warning(f"No matches found for '{stock_search}'")
            stock_options = stock_names[:30]  # Show first 30 as fallback
        else:
            stock_options = [f"{s[0]}: {s[1]}" for s in filtered_stocks]
    else:
        stock_options = stock_names[:30]  # Show first 30 by default
    
    # Stock selector
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=stock_options,
        index=0
    )
    
    # Extract symbol from selection
    symbol = selected_stock.split(":")[0].strip()
    
    # Option to enable multi-timeframe analysis (default on)
    use_multi_timeframe = st.sidebar.checkbox("Enable Multi-Timeframe Analysis", value=True)
    
    # Auto refresh option
    refresh_options = st.sidebar.radio(
        "Auto Refresh Data", 
        ["None", "Every 1 minute", "Every 5 minutes", "Every 25 minutes"],
        index=0
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get comprehensive data (daily + intraday + real-time)
        with st.spinner("Fetching comprehensive data..."):
            data = create_comprehensive_dataset(symbol, selected_timeframe)
            
            # ENHANCEMENT 1: Get multi-timeframe data if enabled
            multi_timeframe_data = None
            if use_multi_timeframe:
                with st.spinner("Analyzing multiple timeframes..."):
                    multi_timeframe_data = fetch_multi_timeframe_data(symbol, selected_timeframe)
        
        if not data.empty:
            # Display data source information
            last_idx = data.index[-1]
            data_source = data.loc[last_idx, 'data_source']
            source_display = {
                'daily': 'Historical Daily Data',
                'intraday': 'Intraday Data',
                'intraday_aggregated': 'Aggregated Intraday Data',
                'realtime': 'Real-time Price'
            }
            source_text = source_display.get(data_source, data_source)
            
            st.markdown(f"""
            <div class="data-source-info">
            ℹ️ <b>Latest Data Source:</b> {source_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate indicators with the comprehensive data
            with st.spinner("Calculating indicators..."):
                data_with_indicators = calculate_indicators(data)
            
            # Plot chart
            fig = plot_chart(data_with_indicators, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No data available for {symbol}")
    
    with col2:
        st.subheader("Signal Analysis")
        
        if not data.empty:
            # Generate signal using comprehensive data and multi-timeframe analysis
            signal = generate_signals(data_with_indicators, selected_timeframe, multi_timeframe_data)
            
            # Get current price information
            current_price = data.loc[data.index[-1], 'close']
            
            if signal and 'direction' in signal:
                # Add price data to signal
                signal['last_price'] = current_price
                
                # Display signal with appropriate styling
                if signal['direction'] == 'BUY':
                    st.markdown(f"<p class='buy-signal'>🔵 BUY SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                elif signal['direction'] == 'SELL':
                    st.markdown(f"<p class='sell-signal'>🔴 SELL SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='neutral-signal'>⚪ NEUTRAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                
                # ENHANCEMENT 3: Display market regime information
                if 'market_regime' in signal:
                    regime_display = {
                        'trending': 'Trending Market',
                        'ranging': 'Ranging/Consolidation Market',
                        'volatile': 'Volatile Market',
                        'weak_trend': 'Weak Trend',
                        'unknown': 'Unknown Market Regime'
                    }
                    regime_text = regime_display.get(signal['market_regime'], signal['market_regime'])
                    
                    st.markdown(f"""
                    <div class="regime-info">
                    🏛️ <b>Market Regime:</b> {regime_text}
                    </div>
                    """, unsafe_allow_html=True)
                
                # ENHANCEMENT 2: Display signal clarity information
                if 'signal_clarity' in signal:
                    clarity = signal['signal_clarity']
                    agreement_pct = clarity['agreement_ratio'] * 100
                    
                    if clarity['clarity_score'] > 70:
                        clarity_text = "Strong signal clarity with high agreement"
                    elif clarity['clarity_score'] > 50:
                        clarity_text = "Moderate signal clarity"
                    else:
                        clarity_text = "Mixed signals with low clarity"
                    
                    st.markdown(f"""
                    <div class="signal-clarity-info">
                    🔍 <b>Signal Clarity:</b> {clarity['clarity_score']:.0f}/100 ({clarity_text})<br>
                    <b>Indicator Agreement:</b> {agreement_pct:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display conflicts if any
                    if clarity['conflicts'] and len(clarity['conflicts']) > 0:
                        with st.expander("View Signal Conflicts"):
                            for conflict in clarity['conflicts']:
                                st.markdown(f"• **{conflict['direction']}**: {conflict['reason']} ({conflict['points']} points)")
                
                # Dashboard page (continued)
                # ENHANCEMENT 1: Display multi-timeframe alignment (continued)
                    else:
                        st.markdown(f"""
                        <div class="multi-timeframe-info">
                        📊 <b>Multi-timeframe Analysis:</b> No clear alignment across timeframes
                        </div>
                        """, unsafe_allow_html=True)
                
                # ENHANCEMENT 4: Display entry signal timing information
                if 'entry_signals' in signal and signal['entry_signals']:
                    entry = signal['entry_signals']
                    
                    if entry['entry_signal']:
                        entry_type = entry['signal_type'].replace('_', ' ').title()
                        entry_strength = entry['strength'].title()
                        
                        st.markdown(f"""
                        <div class="entry-signal-info">
                        ⏱️ <b>Entry Signal Detected:</b> {entry_type}<br>
                        <b>Signal Strength:</b> {entry_strength}
                        </div>
                        """, unsafe_allow_html=True)
                
                # ENHANCEMENT 5: Display historical performance if available
                if 'historical_performance' in signal and signal['historical_performance']:
                    hist = signal['historical_performance']
                    
                    if hist['total'] >= 3:  # Only show if we have enough data
                        st.markdown(f"""
                        <div class="historical-info">
                        📈 <b>Historical Performance:</b> {hist['win_rate']:.1f}% Win Rate<br>
                        <b>Sample Size:</b> {hist['total']} similar signals
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display data source warning if needed
                if data_source != 'realtime':
                    st.markdown(f"""
                    <div class="price-warning">
                    ⚠️ <b>Data Source Note:</b> Current analysis is based on {source_text.lower()}. 
                    Signal accuracy may improve with real-time price updates during market hours.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display timeframe recommendation
                if 'timeframe' in signal:
                    st.markdown(f"<div class='timeframe-info'>✅ <b>Recommended holding period:</b> {signal['timeframe']}</div>", unsafe_allow_html=True)
                
                # Display reasons
                st.subheader("Signal Reasons:")
                for reason in signal['reasons']:
                    st.write(f"• {reason}")
                
                # Option to execute as paper trade
                if st.button("Execute as Paper Trade"):
                    shares = execute_paper_trade(symbol, signal['direction'], current_price, signal['confidence'], signal['timeframe'])
                    
                    # ENHANCEMENT 5: Store signal key for performance tracking
                    signal_key = generate_signal_key(signal)
                    if signal_key and symbol in st.session_state.paper_portfolio['positions']:
                        st.session_state.paper_portfolio['positions'][symbol]['signal_key'] = signal_key
                    
                    st.success(f"Paper trade executed: {signal['direction']} {shares} shares of {symbol} at ${current_price:.2f}")
                
                # Display signal breakdown with timeframes
                if st.checkbox("Show detailed signal breakdown"):
                    st.subheader("Signal Breakdown:")
                    for s in signal['signals']:
                        if 'timeframe' in s:
                            timeframe_info = f" ({s['timeframe']})"
                        else:
                            timeframe_info = ""
                            
                        if s['type'] == 'BUY':
                            st.markdown(f"<span style='color:green'>• BUY: {s['reason']}{timeframe_info} (+{s['points']} points)</span>", unsafe_allow_html=True)
                        elif s['type'] == 'SELL':
                            st.markdown(f"<span style='color:red'>• SELL: {s['reason']}{timeframe_info} (+{s['points']} points)</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:orange'>• NEUTRAL: {s['reason']}{timeframe_info} (+{s['points']} points)</span>", unsafe_allow_html=True)
                
                # Display latest price data
                st.subheader("Latest Price Data:")
                st.write(f"Open: ${data.iloc[-1]['open']:.2f}")
                st.write(f"High: ${data.iloc[-1]['high']:.2f}")
                st.write(f"Low: ${data.iloc[-1]['low']:.2f}")
                st.write(f"Close: ${data.iloc[-1]['close']:.2f}")
                st.write(f"Volume: {int(data.iloc[-1]['volume']):,}")
                st.write(f"Date: {data.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"Data Source: {source_text}")
                
                # Display key indicator values
                st.subheader("Key Indicators:")
                if 'rsi' in data_with_indicators.columns:
                    rsi_value = data_with_indicators['rsi'].iloc[-1]
                    color = "green" if rsi_value < 30 else "red" if rsi_value > 70 else "black"
                    st.markdown(f"RSI: <span style='color:{color}'>{rsi_value:.2f}</span>", unsafe_allow_html=True)
                
                if 'macd' in data_with_indicators.columns:
                    macd_value = data_with_indicators['macd'].iloc[-1]
                    color = "green" if macd_value > 0 else "red"
                    st.markdown(f"MACD: <span style='color:{color}'>{macd_value:.2f}</span>", unsafe_allow_html=True)
                
                if 'macd_signal' in data_with_indicators.columns:
                    st.write(f"MACD Signal: {data_with_indicators['macd_signal'].iloc[-1]:.2f}")
                
                if 'adx' in data_with_indicators.columns:
                    adx_value = data_with_indicators['adx'].iloc[-1]
                    color = "green" if adx_value > 25 else "black"
                    st.markdown(f"ADX: <span style='color:{color}'>{adx_value:.2f}</span>", unsafe_allow_html=True)
                
                # Bollinger Band position
                if 'bb_pct_b' in data_with_indicators.columns:
                    bb_value = data_with_indicators['bb_pct_b'].iloc[-1]
                    color = "red" if bb_value > 0.9 else "green" if bb_value < 0.1 else "black"
                    st.markdown(f"BB Position: <span style='color:{color}'>{bb_value:.2f}</span>", unsafe_allow_html=True)
            else:
                st.info("No clear signals detected")
        else:
            st.error("Cannot generate signals without data")
    
    # Auto refresh logic
    if refresh_options == "Every 1 minute":
        st.write("Page will refresh in 1 minute")
        time.sleep(60)
        st.rerun()
    elif refresh_options == "Every 5 minutes":
        st.write("Page will refresh in 5 minutes")
        time.sleep(300)
        st.rerun()
    elif refresh_options == "Every 25 minutes":
        st.write("Page will refresh in 25 minutes")
        time.sleep(1500)  # 25 minutes = 1500 seconds
        st.rerun()

# Scanner page
def show_scanner():
    st.title("🔍 Enhanced Stock Signal Scanner")
    
    # Initialize paper trading
    initialize_paper_trading()
    
    # Settings
    st.sidebar.header("Scanner Settings")
    
    # Trading timeframe selection
    timeframe = st.sidebar.radio(
        "Trading Timeframe",
        ["Short-term (1-7 days)", "Medium-term (1-4 weeks)", "Long-term (1-3 months)"],
        index=0  # Default to short-term
    )
    
    # Map display timeframe to internal timeframe code
    timeframe_map = {
        "Short-term (1-7 days)": "short",
        "Medium-term (1-4 weeks)": "medium",
        "Long-term (1-3 months)": "long"
    }
    selected_timeframe = timeframe_map[timeframe]
    
    # Choose scan mode
    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (Top Stocks)", "Custom Scan"]
    )
    
    # Minimum confidence threshold
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 
        50, 95, 65
    )
    
    # Filter by signal direction
    signal_direction = st.sidebar.radio(
        "Signal Direction",
        ["All", "Buy Only", "Sell Only"]
    )
    
    # Data completeness options
    use_comprehensive_data = st.sidebar.checkbox("Use Comprehensive Data", value=True, help="Combines daily, intraday, and real-time data for more accurate signals")
    
    # Enable multi-timeframe analysis
    use_multi_timeframe = st.sidebar.checkbox("Enable Multi-Timeframe Analysis", value=True, help="Analyzes multiple timeframes for signal confirmation")
    
    # Market regime filter
    market_regime_filter = st.sidebar.multiselect(
        "Filter by Market Regime",
        ["trending", "ranging", "volatile", "weak_trend", "unknown"],
        default=[]
    )
    
    # Signal clarity filter
    min_clarity_score = st.sidebar.slider(
        "Minimum Signal Clarity", 
        0, 100, 40
    )
    
    # Stocks to scan
    stocks_to_scan = []
    
    if scan_mode == "Quick Scan (Top Stocks)":
        # Get top 10 major stocks for quick scan
        top_stocks = [s[0] for s in st.session_state.stock_universe[:10]]
        stocks_to_scan = top_stocks
        st.write(f"Scanning {len(stocks_to_scan)} top US stocks...")
        st.warning("Alpha Vantage free tier limits: 25 API calls per day. Scanning will use 10 of your daily quota.")
    else:
        # Custom stock input section
        st.sidebar.subheader("Select Custom Stocks")
        
        # Create a search box for adding stocks
        stock_search = st.sidebar.text_input("Search for stocks to add:")
        
        if stock_search:
            # Filter stocks based on search
            filtered_stocks = [(s[0], s[1]) for s in st.session_state.stock_universe 
                              if stock_search.upper() in s[0] or stock_search.lower() in s[1].lower()]
            
            if not filtered_stocks:
                st.sidebar.warning(f"No matches found for '{stock_search}'")
            else:
                options = [f"{s[0]}: {s[1]}" for s in filtered_stocks[:20]]  # Limit to 20 results
                selected_stocks = st.sidebar.multiselect(
                    "Select stocks to scan:",
                    options=options
                )
                
                # Extract symbols from selections
                if selected_stocks:
                    stocks_to_scan = [s.split(":")[0].strip() for s in selected_stocks]
                    st.write(f"Scanning {len(stocks_to_scan)} custom stocks...")
                    if len(stocks_to_scan) > 10:
                        st.warning(f"Scanning {len(stocks_to_scan)} stocks will use {len(stocks_to_scan)} of your daily 25 API calls quota.")
        
        # Default options if no search or selection
        if not stocks_to_scan:
            default_options = [f"{s[0]}: {s[1]}" for s in st.session_state.stock_universe[:20]]
            st.sidebar.write("Or select from popular stocks:")
            selected_default = st.sidebar.multiselect(
                "Popular stocks:",
                options=default_options
            )
            
            if selected_default:
                stocks_to_scan = [s.split(":")[0].strip() for s in selected_default]
                st.write(f"Scanning {len(stocks_to_scan)} stocks...")
                if len(stocks_to_scan) > 10:
                    st.warning(f"Scanning {len(stocks_to_scan)} stocks will use {len(stocks_to_scan)} of your daily 25 API calls quota.")
    
    # Display stock list
    if stocks_to_scan:
        with st.expander("View stocks to scan"):
            st.write(", ".join(stocks_to_scan))
        
        # Scan button
        if st.button("Run Scan"):
            with st.spinner(f"Scanning for {timeframe} signals... (This may take a few minutes due to API rate limits)"):
                # Use enhanced signal generation with real-time prices if requested
                if use_comprehensive_data:
                    signals = generate_enhanced_signals(stocks_to_scan, min_confidence, selected_timeframe)
                else:
                    signals = scan_for_signals(stocks_to_scan, min_confidence, selected_timeframe)
                
                # Apply filters
                if signal_direction == "Buy Only":
                    signals = [s for s in signals if s['direction'] == 'BUY']
                elif signal_direction == "Sell Only":
                    signals = [s for s in signals if s['direction'] == 'SELL']
                
                # Apply market regime filter if selected
                if market_regime_filter:
                    signals = [s for s in signals if 'market_regime' in s and s['market_regime'] in market_regime_filter]
                
                # Apply signal clarity filter
                if min_clarity_score > 0:
                    signals = [s for s in signals if 'signal_clarity' in s and s['signal_clarity']['clarity_score'] >= min_clarity_score]
                
                if signals:
                    # Store signals in session state
                    st.session_state.last_signals = signals
                    # Save signals to disk with the paper trading data
                    save_paper_trading_data()
                    
                    st.success(f"Found {len(signals)} signals!")
                    
                    # Option to auto-execute paper trades
                    auto_execute = st.checkbox("Auto-execute these signals as paper trades", value=False)
                    
                    if auto_execute:
                        for signal in signals:
                            symbol = signal['symbol']
                            direction = signal['direction']
                            price = signal['last_price']
                            confidence = signal['confidence']
                            timeframe = signal.get('timeframe', 'Medium-term')
                            
                            # Execute the paper trade
                            shares = execute_paper_trade(symbol, direction, price, confidence, timeframe)
                            
                            # ENHANCEMENT 5: Store signal key for performance tracking
                            signal_key = generate_signal_key(signal)
                            if signal_key and symbol in st.session_state.paper_portfolio['positions']:
                                st.session_state.paper_portfolio['positions'][symbol]['signal_key'] = signal_key
                            
                            st.success(f"Paper trade executed: {direction} {shares} shares of {symbol} at ${price:.2f}")
                    
                    # Display signals in a table first (summary)
                    signal_data = []
                    for signal in signals:
                        timeframe_str = signal.get('timeframe', 'Medium-term')
                        data_source = signal.get('data_source', 'unknown')
                        
                        # Build price display with source info
                        price_display = f"${signal['last_price']:.2f}"
                        if 'prev_close' in signal and 'price_diff_pct' in signal:
                            price_diff = signal['price_diff_pct']
                            if abs(price_diff) > 1.0:  # If difference is significant
                                price_display += f" (Δ {price_diff:.1f}%)"
                        
                        # Add market regime info
                        regime = signal.get('market_regime', 'unknown')
                        
                        # Add signal clarity info
                        clarity_score = "N/A"
                        if 'signal_clarity' in signal:
                            clarity_score = f"{signal['signal_clarity']['clarity_score']:.0f}/100"
                        
                        signal_data.append({
                            'Symbol': signal['symbol'],
                            'Direction': signal['direction'],
                            'Confidence': f"{signal['confidence']}%",
                            'Last Price': price_display,
                            'Market Regime': regime.capitalize(),
                            'Signal Clarity': clarity_score,
                            'Timeframe': timeframe_str,
                            'Data Source': data_source
                        })
                    
                    df_signals = pd.DataFrame(signal_data)
                    st.dataframe(df_signals, use_container_width=True)
                    
                    # Display detailed signals
                    st.subheader("Detailed Signals")
                    for i, signal in enumerate(signals, 1):
                        if signal['direction'] == 'BUY':
                            st.markdown(f"""
                            ### 🔵 BUY - {signal['symbol']} - {signal['confidence']}% Confidence
                            """)
                        else:
                            st.markdown(f"""
                            ### 🔴 SELL - {signal['symbol']} - {signal['confidence']}% Confidence
                            """)
                        
                        # Display data source information
                        data_source = signal.get('data_source', 'unknown')
                        source_display = {
                            'daily': 'Historical Daily Data',
                            'intraday': 'Intraday Data',
                            'intraday_aggregated': 'Aggregated Intraday Data',
                            'realtime': 'Real-time Price'
                        }
                        source_text = source_display.get(data_source, data_source)
                        st.markdown(f"""
                        <div class="data-source-info">
                        ℹ️ <b>Data Source:</b> {source_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ENHANCEMENT 3: Display market regime information
                        if 'market_regime' in signal:
                            regime_display = {
                                'trending': 'Trending Market',
                                'ranging': 'Ranging/Consolidation Market',
                                'volatile': 'Volatile Market',
                                'weak_trend': 'Weak Trend',
                                'unknown': 'Unknown Market Regime'
                            }
                            regime_text = regime_display.get(signal['market_regime'], signal['market_regime'])
                            
                            st.markdown(f"""
                            <div class="regime-info">
                            🏛️ <b>Market Regime:</b> {regime_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ENHANCEMENT 2: Display signal clarity information
                        if 'signal_clarity' in signal:
                            clarity = signal['signal_clarity']
                            agreement_pct = clarity['agreement_ratio'] * 100
                            
                            if clarity['clarity_score'] > 70:
                                clarity_text = "Strong signal clarity with high agreement"
                            elif clarity['clarity_score'] > 50:
                                clarity_text = "Moderate signal clarity"
                            else:
                                clarity_text = "Mixed signals with low clarity"
                            
                            st.markdown(f"""
                            <div class="signal-clarity-info">
                            🔍 <b>Signal Clarity:</b> {clarity['clarity_score']:.0f}/100 ({clarity_text})<br>
                            <b>Indicator Agreement:</b> {agreement_pct:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ENHANCEMENT 1: Display multi-timeframe alignment if available
                        if 'multi_timeframe_alignment' in signal and signal['multi_timeframe_alignment']:
                            mtf = signal['multi_timeframe_alignment']
                            
                            if mtf['aligned']:
                                alignment_direction = "Bullish" if mtf['direction'] == 'bullish' else "Bearish"
                                alignment_score = mtf['score'] * 100
                                
                                st.markdown(f"""
                                <div class="multi-timeframe-info">
                                📊 <b>Multi-timeframe Alignment:</b> {alignment_direction} across {mtf['timeframes_analyzed']} timeframes<br>
                                <b>Alignment Strength:</b> {alignment_score:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # ENHANCEMENT 4: Display entry signal timing information
                        if 'entry_signals' in signal and signal['entry_signals']:
                            entry = signal['entry_signals']
                            
                            if entry['entry_signal']:
                                entry_type = entry['signal_type'].replace('_', ' ').title()
                                entry_strength = entry['strength'].title()
                                
                                st.markdown(f"""
                                <div class="entry-signal-info">
                                ⏱️ <b>Entry Signal Detected:</b> {entry_type}<br>
                                <b>Signal Strength:</b> {entry_strength}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display price comparison warning if significant difference
                        if 'prev_close' in signal and 'price_diff_pct' in signal and abs(signal['price_diff_pct']) > 1.0:
                            st.markdown(f"""
                            <div class="price-warning">
                            ⚠️ <b>Price Movement:</b> Current price (${signal['last_price']:.2f}) differs from previous close (${signal['prev_close']:.2f}) 
                            by {signal['price_diff_pct']:.2f}%. {signal.get('confidence_adjustment', '')} confidence.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display recommended timeframe
                        if 'timeframe' in signal:
                            st.markdown(f"<div class='timeframe-info'>✅ <b>Recommended holding period:</b> {signal['timeframe']}</div>", unsafe_allow_html=True)
                        
                        # Display price information
                        st.write(f"Current Price: ${signal['last_price']:.2f}")
                        if 'prev_close' in signal:
                            st.write(f"Previous Close: ${signal['prev_close']:.2f} on {signal.get('prev_date', 'previous trading day')}")
                        
                        # Display reasons
                        st.subheader("Signal Reasons:")
                        for reason in signal['reasons']:
                            st.write(f"• {reason}")
                        
                        # Add button to execute individual trade
                        if st.button(f"Execute {signal['direction']} for {signal['symbol']}", key=f"btn_{signal['symbol']}"):
                            shares = execute_paper_trade(
                                signal['symbol'], 
                                signal['direction'],
                                signal['last_price'],
                                signal['confidence'],
                                signal.get('timeframe', 'Medium-term')
                            )
                            
                            # ENHANCEMENT 5: Store signal key for performance tracking
                            signal_key = generate_signal_key(signal)
                            if signal_key and signal['symbol'] in st.session_state.paper_portfolio['positions']:
                                st.session_state.paper_portfolio['positions'][signal['symbol']]['signal_key'] = signal_key
                            
                            st.success(f"Paper trade executed: {signal['direction']} {shares} shares of {signal['symbol']} at ${signal['last_price']:.2f}")
                        
                        # Add a separator between signals
                        if i < len(signals):
                            st.markdown("---")
                    
                    # Option to save results
                    if st.button("Save Results to CSV"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        df_signals.to_csv(f"signals_{timestamp}.csv", index=False)
                        st.success(f"Saved results to signals_{timestamp}.csv")
                else:
                    st.info(f"No {timeframe} signals found matching your criteria")
    else:
        st.warning("Please select at least one stock to scan")

# Real-time Monitor
def show_real_time_monitor():
    st.title("📊 Real-time Market Monitor")
    
    # Settings
    st.sidebar.header("Real-time Settings")
    
    # Get stock symbols from the loaded universe
    symbol_options = [s[0] for s in st.session_state.stock_universe[:100]]  # First 100 stocks
    
    # Symbol selection
    monitoring_symbols = st.sidebar.multiselect(
        "Select symbols to monitor",
        options=symbol_options,
        default=["AAPL", "MSFT", "AMZN"]
    )
    
    # Update websocket symbols if changed
    if 'ws_symbols' not in st.session_state or st.session_state.ws_symbols != monitoring_symbols:
        st.session_state.ws_symbols = monitoring_symbols
        
        # Reset websocket connection if already running
        if 'ws_running' in st.session_state and st.session_state.ws_running:
            try:
                st.session_state.ws.close()
            except:
                pass
            st.session_state.ws_running = False
    
    # Monitor paper trading positions too
    show_paper_positions = st.sidebar.checkbox("Show Paper Trading Positions", value=True)
    
    # Set up real-time monitoring
    setup_real_time_monitoring(monitoring_symbols)
    
    # Show current quotes for selected symbols
    st.subheader("Current Quotes")
    
    quotes_data = []
    for symbol in monitoring_symbols:
        try:
            # Get quote data
            quote = get_current_quote(symbol)
            if quote and 'c' in quote:
                change = quote['c'] - quote['pc']
                change_pct = (change / quote['pc']) * 100
                
                quotes_data.append({
                    'Symbol': symbol,
                    'Price': quote['c'],
                    'Change': change,
                    'Change %': f"{change_pct:.2f}%",
                    'High': quote['h'],
                    'Low': quote['l'],
                    'Open': quote['o'],
                    'Prev Close': quote['pc']
                })
        except Exception as e:
            st.error(f"Error fetching quote for {symbol}: {e}")
    
    if quotes_data:
        # Convert to DataFrame for display
        df_quotes = pd.DataFrame(quotes_data)
        
        # Style the DataFrame with colors for positive/negative changes
        def color_change(val):
            try:
                val_str = str(val)
                if '%' in val_str:
                    val_num = float(val_str.strip('%'))
                    color = 'green' if val_num >= 0 else 'red'
                    return f'color: {color}'
                elif isinstance(val, (int, float)):
                    color = 'green' if val >= 0 else 'red'
                    return f'color: {color}'
            except:
                return ''
            return ''
        
        # Apply styling and display
        try:
            st.dataframe(df_quotes.style.applymap(color_change, subset=['Change', 'Change %']), use_container_width=True)
        except Exception as e:
            st.error(f"Error styling dataframe: {e}")
            st.dataframe(df_quotes, use_container_width=True)
        
        # Update paper trading positions
        if show_paper_positions and st.session_state.paper_portfolio['positions']:
            st.subheader("Paper Trading Positions")
            
            # Convert quotes_data to dict for update_paper_positions
            quotes_dict = {row['Symbol']: row['Price'] for row in quotes_data}
            
            # Update positions with latest prices
            portfolio = update_paper_positions(quotes_dict)
            
            # Display paper positions
            positions_data = []
            for symbol, position in portfolio['positions'].items():
                positions_data.append({
                    'Symbol': symbol,
                    'Direction': position['direction'],
                    'Shares': position['shares'],
                    'Entry Price': f"${position['entry_price']:.2f}",
                    'Current Price': f"${position.get('current_price', position['entry_price']):.2f}",
                    'P&L': f"${position.get('pnl', 0):.2f}",
                    'P&L %': f"{position.get('pnl_pct', 0):.2f}%"
                })
            
            if positions_data:
                df_positions = pd.DataFrame(positions_data)
                try:
                    st.dataframe(df_positions.style.applymap(color_pnl), use_container_width=True)
                except Exception as e:
                    st.error(f"Error styling positions dataframe: {e}")
                    st.dataframe(df_positions, use_container_width=True)
            else:
                st.info("No open paper trading positions")
    
    # Add auto-refresh options
    refresh_options = st.radio(
        "Auto-refresh", 
        ["None", "Every 5 seconds", "Every minute", "Every 25 minutes"],
        index=0,
        horizontal=True
    )
    
    if refresh_options == "Every 5 seconds":
        time.sleep(5)
        st.rerun()
    elif refresh_options == "Every minute":
        time.sleep(60)
        st.rerun()
    elif refresh_options == "Every 25 minutes":
        time.sleep(1500)  # 25 minutes
        st.rerun()

# Documentation page (continued)
def show_documentation():
    st.title("📚 Enhanced Technical Trading Signal System")
    
    st.markdown("""
    This dashboard analyzes stocks using technical indicators to generate trading signals with confidence scores and recommended timeframes.
    """)
    
    with st.expander("📋 System Overview", expanded=True):
        st.markdown("""
        ### System Architecture
        
        Our enhanced technical analysis system uses a comprehensive data approach:
        
        1. **Historical Daily Data**: From Alpha Vantage API for previous trading days
        2. **Intraday Data**: From Alpha Vantage API for the current trading day
        3. **Real-time Data**: From Finnhub API for up-to-the-minute price updates
        4. **Technical Analysis**: 8 key indicators to generate signals
        5. **Signal Confidence**: Weighted scoring system (50-95%)
        6. **Timeframe Analysis**: Categorizes optimal holding periods
        7. **Paper Trading**: Virtual portfolio to test signals without real money
        8. **Data Integration**: Seamless combination of all data sources for accurate analysis
        
        ### New Enhanced Features
        
        1. **Multi-Timeframe Alignment**: Analyzes multiple timeframes to ensure trends align
        2. **Signal Clarity & Conflict Resolution**: Measures the agreement between indicators
        3. **Market Regime Detection**: Adapts to ranging, trending, or volatile markets
        4. **Entry Timing System**: Identifies optimal entry points on shorter timeframes
        5. **Statistical Performance Tracking**: Learns from historical signal performance
        
        The system requires a minimum 65% confidence score to display trading signals.
        """)
    
    with st.expander("📊 Data Integration Approach"):
        st.markdown("""
        ### Comprehensive Data Integration
        
        Our enhanced system now uses a three-tier approach to ensure complete data coverage:
        
        1. **Historical Daily Data (Alpha Vantage)**
           - Provides end-of-day OHLCV data for previous trading days
           - Used for long-term trend analysis and historical context
           - Daily bars going back 30-200 days depending on timeframe
        
        2. **Intraday Data (Alpha Vantage)**
           - Fills the critical gap between the previous day's close and current time
           - Provides price action for the current trading day
           - 15-minute interval data used to create a daily composite for today
        
        3. **Real-time Price Updates (Finnhub)**
           - Provides the most current price for up-to-the-second accuracy
           - Used to update the latest close price in the dataset
           - Enables trading decisions based on the most current market conditions
        
        This three-tier approach ensures there are no gaps in your data, leading to more accurate technical indicators and signal generation.
        """)
    
    with st.expander("⏱️ Trading Timeframes"):
        st.markdown("""
        ### Signal Timeframes
        
        The system categorizes signals by their optimal holding period:
        
        | Timeframe | Holding Period | Indicators |
        |-----------|----------------|------------|
        | Very Short-term | 1-3 days | Stochastic RSI |
        | Short-term | 3-10 days | RSI, Bollinger Bands |
        | Medium-term | 2-4 weeks | MACD, EMA Cloud |
        | Long-term | 1-3 months | EMA50 vs EMA200 |
        
        The overall timeframe recommendation is based on the strongest signals present.
        """)
    
    with st.expander("🔄 Multi-Timeframe Analysis"):
        st.markdown("""
        ### Multi-Timeframe Alignment System
        
        The multi-timeframe analysis system examines the same stock across different timeframes to ensure trend alignment:
        
        | Main Trading Timeframe | Timeframes Analyzed |
        |------------------------|---------------------|
        | Short-term (1-7 days) | 15min, 1h, Daily |
        | Medium-term (1-4 weeks) | 1h, Daily, Weekly |
        | Long-term (1-3 months) | Daily, Weekly, Monthly |
        
        #### How It Works
        
        1. **Indicator Agreement**: Checks if the same indicators show the same direction across timeframes
        2. **Alignment Score**: Calculates a 0-100% score based on how well timeframes align
        3. **Confidence Boost**: Adds up to 10 points to signal confidence when timeframes align
        4. **Conflict Detection**: Warns when higher timeframes contradict the main timeframe
        
        This approach significantly reduces false signals by ensuring you're trading in the direction of the larger trends.
        """)
    
    with st.expander("🧠 Market Regime Detection"):
        st.markdown("""
        ### Market Regime Detection System
        
        The system now automatically detects the current market regime and adapts indicator weights accordingly:
        
        | Regime | Characteristics | Best Indicators |
        |--------|-----------------|-----------------|
        | Trending | ADX > 25, clear directional movement | MACD, EMAs, ADX |
        | Ranging | Tight price consolidation, low ADX | RSI, Bollinger Bands, Stochastic |
        | Volatile | High price variance, erratic moves | Volume, ATR, Bollinger width |
        | Weak Trend | Slight directionality, ADX 15-25 | Combination of trend and range |
        
        #### How It Works
        
        1. **Detection Metrics**: Uses ADX, Bollinger Band width, price volatility, and MA angles
        2. **Dynamic Weighting**: Adjusts the importance of each indicator based on the detected regime
        3. **Signal Targeting**: Focuses on range-bound signals in ranging markets, trend signals in trending markets
        
        This adaptive approach ensures you're using the right tools for the current market conditions.
        """)
    
    with st.expander("🔍 Signal Clarity & Conflict Resolution"):
        st.markdown("""
        ### Signal Clarity System
        
        The signal clarity system measures how strongly indicators agree with each other:
        
        #### Clarity Metrics
        
        - **Agreement Ratio**: Percentage of indicators agreeing on direction (BUY or SELL)
        - **Clarity Score**: 0-100 score based on agreement level and conflicting signals
        - **Conflict Identification**: Lists specific indicators that contradict the main signal
        
        #### How It Affects Confidence
        
        - **High Clarity (75-100)**: Boosts confidence score (+5 points max)
        - **Medium Clarity (40-75)**: No adjustment
        - **Low Clarity (<40)**: Reduces confidence score (-10 points max)
        
        This system helps you focus on high-conviction signals where multiple indicators point in the same direction.
        """)
    
    with st.expander("⏱️ Entry Timing System"):
        st.markdown("""
        ### Entry Timing System
        
        Once a trading signal is generated, the entry timing system helps you pinpoint optimal entry points:
        
        #### For BUY Signals
        
        - MACD crossing above signal line
        - RSI bouncing from oversold (<30)
        - Price crossing above short-term EMA
        - Bollinger Band bounce from lower band
        
        #### For SELL Signals
        
        - MACD crossing below signal line
        - RSI dropping from overbought (>70)
        - Price crossing below short-term EMA
        - Bollinger Band breakdown from upper band
        
        #### Signal Strength Classification
        
        - **Strong**: High-probability entry signals (RSI extremes, BB touches)
        - **Moderate**: Medium-probability signals (MA crossovers)
        
        This system prevents late entries and helps you time your trades more effectively.
        """)
    
    with st.expander("📊 Statistical Performance Tracking"):
        st.markdown("""
        ### Signal Performance Tracking
        
        The system now tracks historical performance of each signal type:
        
        #### How It Works
        
        1. **Signal Fingerprinting**: Creates a unique key for each signal pattern
        2. **Win/Loss Tracking**: Records outcomes when trades are closed
        3. **Performance Metrics**: Win rate, sample size, average return
        4. **Confidence Adjustment**: Boosts confidence for historically successful patterns, reduces for poor performers
        
        #### Benefits
        
        - System learns from its own performance
        - Higher confidence in signals with proven track records
        - Helps identify which indicator combinations work best
        
        This creates a self-improving system that gets smarter with each trade.
        """)
    
    with st.expander("📝 Paper Trading System"):
        st.markdown("""
        ### Paper Trading Features
        
        The paper trading system allows you to test trading signals in a risk-free environment:
        
        - **Starting Capital**: $100,000 virtual money
        - **Position Sizing**: Based on signal confidence (2-5% of portfolio)
        - **Real-time P&L**: Updated continuously using market data
        - **Trade History**: Records all closed positions with performance metrics
        - **Performance Analytics**: Win rate, average return, total P&L
        - **Data Persistence**: Your portfolio data is saved automatically between sessions
        - **Signal Performance Tracking**: Stores which signal types produce the best results
        
        You can execute trades manually from the dashboard or scanner, or set it to automatically execute new signals.
        """)
        
    with st.expander("🔄 Real-time Data Integration"):
        st.markdown("""
        ### Seamless Data Integration
        
        The system now offers seamless integration between data sources:
        
        - **Historical** → **Intraday** → **Real-time**: Data flows naturally between timeframes
        - **Source Tracking**: Each price point is tagged with its data source
        - **Confidence Adjustment**: Signal confidence is adjusted based on data recency and source
        - **Price Movement Analysis**: Significant price moves between data sources are highlighted
        - **Visualization**: Comprehensive chart integrating all data sources
        
        This approach ensures technical indicators are always calculated with the most complete dataset available.
        """)
    
    with st.expander("📘 API Usage Information"):
        st.markdown("""
        ### API Limitations & Best Practices
        
        #### Alpha Vantage (Historical & Intraday Data)
        - Free tier limit: 5 API calls per minute, 500 per day
        - Used for: Historical price data, daily OHLCV data, intraday data
        - Intraday data is updated at the end of each trading day in the free tier
        - Premium tier available for higher limits and real-time intraday data
        
        #### Finnhub (Real-time Data)
        - Free tier includes: Websocket access for real-time trades
        - Used for: Real-time trade monitoring, current quotes
        - Premium tier required for historical OHLCV data
        
        #### Best Practices
        - Limit scanning to 5-10 stocks at a time
        - Allow 60 seconds between scans to respect API limits
        - Use the dashboard for detailed analysis of individual stocks
        - Use scanner sparingly to find potential opportunities
        - Save API calls by using cached data when possible
        """)
    
    with st.expander("🏆 Best Practices for Trading"):
        st.markdown("""
        ### How to Use This System Effectively
        
        #### Best Practices
        
        1. **Multi-Timeframe Alignment**: Only take trades where multiple timeframes agree
        
        2. **Signal Clarity Focus**: Prioritize signals with high clarity scores (>70)
        
        3. **Market Regime Awareness**: Use trend indicators in trending markets, oscillators in ranging markets
        
        4. **Entry Timing**: Wait for an entry signal after a main signal is detected
        
        5. **Signal Performance**: Pay attention to the historical performance of signal types
        
        6. **Risk Management**: Always use stop-losses (typically 5-7% for swing trades)
        
        7. **Position Sizing**: Let confidence score guide position size (higher confidence = larger position)
        
        #### Risk Management Guidelines
        
        - Position sizing: 1-2% account risk per trade
        - Stop loss placement: Below recent support for buys, above recent resistance for sells
        - Take profits: Use the recommended timeframe as a guide for exit strategy
        """)
    
    with st.expander("💾 Data Persistence Features"):
        st.markdown("""
        ### Data Persistence
        
        This application includes data persistence to save your trading data between sessions:
        
        1. **Paper Trading Portfolio**: Your paper trading portfolio, including positions, trade history, and performance metrics, is automatically saved
        
        2. **Trading Signals**: The last set of signals from the scanner is preserved between sessions
        
        3. **API Keys**: Your API keys are securely stored and loaded when the application starts
        
        4. **Signal Performance History**: Historical performance of different signal types is tracked and saved
        
        Data is saved in the following files:
        - `paper_trading_data.pkl`: Contains your paper trading portfolio data and signal performance
        - `api_keys.json`: Contains your API keys for Alpha Vantage and Finnhub
        
        You can manually save your data at any time using the "Save Portfolio Data" button in the Paper Trading section.
        """)
    
    with st.expander("⏲️ Auto-Refresh Options"):
        st.markdown("""
        ### Auto-Refresh Features
        
        The application includes multiple auto-refresh options to keep your data current:
        
        1. **Dashboard Refresh**: Options to refresh every 1 minute, 5 minutes, or 25 minutes
        
        2. **Real-time Monitor Refresh**: Options for 5-second, 1-minute, or 25-minute intervals
        
        3. **Paper Trading Refresh**: Options to update your portfolio values automatically
        
        The 25-minute refresh option is particularly useful for managing API rate limits with Alpha Vantage,
        as it allows you to refresh just before the hourly quota resets.
        """)

if __name__ == "__main__":
    main()
