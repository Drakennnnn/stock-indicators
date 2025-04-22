import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import finnhub
import requests
import time
from datetime import datetime, timedelta
import pytz
import websocket
import json
import threading
import queue
from io import StringIO
import csv

# Set page configuration
st.set_page_config(
    page_title="Advanced Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API keys
FINNHUB_API_KEY = "d03bkkpr01qvvb93ems0d03bkkpr01qvvb93emsg"  # For real-time data
ALPHA_VANTAGE_API_KEY = "BY8DWVP73ZRGRGWO"  # For historical data

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

# Initialize session state for stock universe if not exists
if 'stock_universe' not in st.session_state:
    st.session_state.stock_universe = None

# Initialize paper trading session state
def initialize_paper_trading():
    if 'paper_portfolio' not in st.session_state:
        st.session_state.paper_portfolio = {
            'cash': 100000,  # Starting with $100k
            'positions': {},  # Will store {symbol: {'shares': qty, 'entry_price': price, 'entry_time': time, 'signal_confidence': conf}}
            'trade_history': [],  # Will store closed trades
            'performance': {'total_return': 0, 'win_rate': 0, 'trades': 0}
        }
    
    if 'last_signals' not in st.session_state:
        st.session_state.last_signals = []  # Store last scanned signals to avoid duplicates

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
    except Exception as e:
        st.sidebar.error(f"❌ Alpha Vantage API connection failed: {e}")

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

# Function to fetch historical data from Alpha Vantage
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
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from Alpha Vantage: {e}")
        return pd.DataFrame()

# Function to get current quote from Finnhub (real-time)
def get_current_quote(symbol):
    try:
        quote = finnhub_client.quote(symbol)
        return quote
    except Exception as e:
        st.warning(f"Error fetching current quote for {symbol}: {e}")
        return None

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

# Generate trading signals
def generate_signals(df, preferred_timeframe="short"):
    if df.empty or len(df) < 20:
        return None
    
    signals = []
    confidence_score = 50  # Base confidence score
    
    # Indicator weights based on timeframe preference
    weights = {
        "short": {
            "macd_crossover": 10,
            "ema_cloud": 15,
            "ema_trend": 5,
            "rsi": 20,
            "stoch_rsi": 15,
            "bb": 15,
            "volume": 15,
            "adx": 5
        },
        "medium": {
            "macd_crossover": 15,
            "ema_cloud": 10,
            "ema_trend": 10,
            "rsi": 15,
            "stoch_rsi": 10,
            "bb": 10,
            "volume": 15,
            "adx": 15
        },
        "long": {
            "macd_crossover": 15,
            "ema_cloud": 5,
            "ema_trend": 20,
            "rsi": 10,
            "stoch_rsi": 5,
            "bb": 5,
            "volume": 15,
            "adx": 25
        }
    }
    
    # Use weights based on preferred timeframe
    w = weights[preferred_timeframe]
    
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
        
        # Determine overall signal direction and confidence
        buy_points = sum(signal['points'] for signal in signals if signal['type'] == 'BUY')
        sell_points = sum(signal['points'] for signal in signals if signal['type'] == 'SELL')
        
        # Calculate final confidence (cap at 95)
        final_confidence = min(95, confidence_score)
        
        # Determine overall timeframe recommendation based on preferred timeframe
        recommended_timeframe = TIMEFRAMES[preferred_timeframe]
        
        # Determine overall signal
        if buy_points > sell_points and final_confidence >= 65:
            overall_signal = {
                'direction': 'BUY',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'BUY' or s['type'] == 'NEUTRAL'],
                'timeframe': recommended_timeframe
            }
        elif sell_points > buy_points and final_confidence >= 65:
            overall_signal = {
                'direction': 'SELL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'SELL' or s['type'] == 'NEUTRAL'],
                'timeframe': recommended_timeframe
            }
        else:
            overall_signal = {
                'direction': 'NEUTRAL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'NEUTRAL'],
                'timeframe': "Wait for clearer signals"
            }
        
        return overall_signal
    
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return None

# Enhanced signal generation with real-time price updates
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
            
            # Step 1: Get historical data from Alpha Vantage for indicators
            df = fetch_alpha_vantage_daily(symbol, timeframe)
            if df.empty:
                continue
            
            # Store the historical price before updating
            historical_price = df['close'].iloc[-1]
            historical_date = df['timestamp'].iloc[-1]
            
            # Step 2: Get real-time price from Finnhub
            quote = get_current_quote(symbol)
            if not quote or 'c' not in quote:
                # If we can't get current quote, use historical price but mark it
                real_time_price = historical_price
                price_warning = True
            else:
                real_time_price = quote['c']
                price_warning = False
                
                # Step 3: Update the last day's close price with real-time data
                # This is crucial for any indicators that use the most recent price
                df.loc[df.index[-1], 'close'] = real_time_price
            
            # Step 4: Calculate indicators with the updated data
            df = calculate_indicators(df)
            
            # Step 5: Generate signal based on updated data
            signal = generate_signals(df, timeframe)
            
            if signal and signal['confidence'] >= min_confidence and signal['direction'] != 'NEUTRAL':
                # Add both prices to the signal data
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                signal['last_price'] = real_time_price
                signal['historical_price'] = historical_price
                signal['historical_date'] = historical_date.strftime("%Y-%m-%d")
                signal['price_diff_pct'] = ((real_time_price - historical_price) / historical_price) * 100
                
                # Adjust confidence if prices have significant difference
                price_diff_abs = abs(signal['price_diff_pct'])
                
                if price_diff_abs > 2.0:  # More than 2% difference
                    # Flag for significant price movement since last close
                    signal['price_warning'] = True
                    
                    # For large moves that oppose the signal direction
                    if (signal['direction'] == 'BUY' and real_time_price > historical_price) or \
                       (signal['direction'] == 'SELL' and real_time_price < historical_price):
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
                'timeframe': timeframe
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
                'timeframe': timeframe
            }
            portfolio['cash'] += shares * price  # Add cash for short selling
    
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
            'timeframe': position.get('timeframe', 'Unknown')
        }
        
        portfolio['trade_history'].append(trade_record)
        
        # Update performance metrics
        portfolio['performance']['trades'] += 1
        portfolio['performance']['total_return'] += pnl
        
        # Calculate win rate
        winning_trades = sum(1 for trade in portfolio['trade_history'] if trade['pnl'] > 0)
        total_trades = len(portfolio['trade_history'])
        portfolio['performance']['win_rate'] = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Remove position
        del positions[symbol]
        
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

# Fixed version of color_pnl function for dataframe styling
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
            title=f'{symbol} - Daily Chart',
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
    
    # Add reset portfolio button
    if st.sidebar.button("Reset Portfolio"):
        st.session_state.paper_portfolio = {
            'cash': 100000,
            'positions': {},
            'trade_history': [],
            'performance': {'total_return': 0, 'win_rate': 0, 'trades': 0}
        }
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
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh portfolio", value=True)
    if auto_refresh:
        time.sleep(5)  # Refresh every 5 seconds
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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Scanner", "Paper Trading", "Real-time Monitor", "Documentation"])
    
    # Display note about APIs
    st.sidebar.markdown("""
    <div class="api-info">
    <strong>Data Sources:</strong><br>
    - Historical data: Alpha Vantage<br>
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
    st.markdown('<p class="big-font">📈 Technical Trading Signal Dashboard</p>', unsafe_allow_html=True)
    
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
    
    # Auto refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", 1, 60, 15)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get historical data from Alpha Vantage
        with st.spinner("Fetching historical data..."):
            data = fetch_alpha_vantage_daily(symbol, selected_timeframe)
        
        if not data.empty:
            # Get real-time price from Finnhub
            quote = get_current_quote(symbol)
            historical_price = data['close'].iloc[-1]
            
            # Display current vs. historical price
            if quote and 'c' in quote:
                real_time_price = quote['c']
                data.loc[data.index[-1], 'close'] = real_time_price  # Update last close with real-time price
                
                price_diff = real_time_price - historical_price
                price_diff_pct = (price_diff / historical_price) * 100
                
                st.sidebar.success(f"Last Price (Finnhub): ${real_time_price:.2f}")
                
                if abs(price_diff_pct) > 0.5:  # If price difference is significant
                    color = "green" if price_diff > 0 else "red"
                    st.sidebar.markdown(f"<p style='color:{color}'>Change from last close: {price_diff:.2f} ({price_diff_pct:.2f}%)</p>", unsafe_allow_html=True)
                    st.sidebar.info(f"Last Close (Alpha Vantage): ${historical_price:.2f} on {data['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
            else:
                # Use historical price if real-time not available
                st.sidebar.info(f"Last Price (Historical): ${historical_price:.2f}")
            
            # Calculate indicators with updated price if available
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
            # Generate signal using data with updated real-time price
            signal = generate_signals(data_with_indicators, selected_timeframe)
            
            # Store real-time price information
            last_price = quote['c'] if quote and 'c' in quote else historical_price
            
            if signal and 'direction' in signal:
                # Add price data to signal
                signal['last_price'] = last_price
                signal['historical_price'] = historical_price
                if quote and 'c' in quote:
                    signal['price_diff_pct'] = ((quote['c'] - historical_price) / historical_price) * 100
                
                # Display signal with appropriate styling
                if signal['direction'] == 'BUY':
                    st.markdown(f"<p class='buy-signal'>🔵 BUY SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                elif signal['direction'] == 'SELL':
                    st.markdown(f"<p class='sell-signal'>🔴 SELL SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='neutral-signal'>⚪ NEUTRAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                
                # Display price comparison warning if significant difference
                if quote and 'c' in quote and abs(signal.get('price_diff_pct', 0)) > 1.0:
                    st.markdown(f"""
                    <div class="price-warning">
                    ⚠️ <b>Price Alert:</b> Current price (${quote['c']:.2f}) differs from historical close (${historical_price:.2f}) 
                    by {signal['price_diff_pct']:.2f}%. Signal may be affected.
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
                    shares = execute_paper_trade(symbol, signal['direction'], last_price, signal['confidence'], signal['timeframe'])
                    st.success(f"Paper trade executed: {signal['direction']} {shares} shares of {symbol} at ${last_price:.2f}")
                
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
                latest = data.iloc[-1]
                st.subheader("Latest Price Data:")
                st.write(f"Open: ${latest['open']:.2f}")
                st.write(f"High: ${latest['high']:.2f}")
                st.write(f"Low: ${latest['low']:.2f}")
                
                # Show both historical and real-time close if available
                if quote and 'c' in quote:
                    st.write(f"Close (Historical): ${historical_price:.2f}")
                    st.write(f"Current Price: ${quote['c']:.2f}")
                else:
                    st.write(f"Close: ${latest['close']:.2f}")
                
                st.write(f"Volume: {int(latest['volume']):,}")
                st.write(f"Date: {latest['timestamp'].strftime('%Y-%m-%d')}")
                
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
    if auto_refresh:
        st.write(f"Page will refresh in {refresh_interval} minutes")
        time.sleep(refresh_interval * 60)
        st.rerun()

# Scanner page
def show_scanner():
    st.title("🔍 Stock Signal Scanner")
    
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
    
    # Real-time price options
    use_real_time = st.sidebar.checkbox("Use Real-time Prices", value=True)
    
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
                if use_real_time:
                    signals = generate_enhanced_signals(stocks_to_scan, min_confidence, selected_timeframe)
                else:
                    signals = scan_for_signals(stocks_to_scan, min_confidence, selected_timeframe)
                
                # Filter by direction if needed
                if signal_direction == "Buy Only":
                    signals = [s for s in signals if s['direction'] == 'BUY']
                elif signal_direction == "Sell Only":
                    signals = [s for s in signals if s['direction'] == 'SELL']
                
                if signals:
                    # Store signals in session state
                    st.session_state.last_signals = signals
                    
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
                            
                            st.success(f"Paper trade executed: {direction} {shares} shares of {symbol} at ${price:.2f}")
                    
                    # Display signals in a table first (summary)
                    signal_data = []
                    for signal in signals:
                        timeframe_str = signal.get('timeframe', 'Medium-term')
                        
                        # Add real-time vs historical price info if available
                        price_display = f"${signal['last_price']:.2f}"
                        if 'historical_price' in signal and 'price_diff_pct' in signal:
                            price_diff = abs(signal['price_diff_pct'])
                            if price_diff > 1.0:  # If difference is significant
                                price_display += f" (Δ {signal['price_diff_pct']:.1f}%)"
                        
                        signal_data.append({
                            'Symbol': signal['symbol'],
                            'Direction': signal['direction'],
                            'Confidence': f"{signal['confidence']}%",
                            'Last Price': price_display,
                            'Timeframe': timeframe_str,
                            'Time': signal['timestamp']
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
                        
                        # Display price comparison warning if significant difference
                        if 'historical_price' in signal and 'price_diff_pct' in signal and abs(signal['price_diff_pct']) > 1.0:
                            st.markdown(f"""
                            <div class="price-warning">
                            ⚠️ <b>Price Alert:</b> Current price (${signal['last_price']:.2f}) differs from historical close (${signal['historical_price']:.2f}) 
                            by {signal['price_diff_pct']:.2f}%. {signal.get('confidence_adjustment', '')} confidence.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display recommended timeframe
                        if 'timeframe' in signal:
                            st.markdown(f"<div class='timeframe-info'>✅ <b>Recommended holding period:</b> {signal['timeframe']}</div>", unsafe_allow_html=True)
                        
                        # Display price information
                        if 'historical_price' in signal:
                            st.write(f"Current Price: ${signal['last_price']:.2f}")
                            st.write(f"Historical Close: ${signal['historical_price']:.2f} on {signal.get('historical_date', 'last trading day')}")
                        else:
                            st.write(f"Current Price: ${signal['last_price']:.2f}")
                        
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
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh quotes", value=True)
    if auto_refresh:
        time.sleep(5)  # Refresh every 5 seconds
        st.rerun()

# Documentation page
def show_documentation():
    st.title("📚 Technical Trading Signal System")
    
    st.markdown("""
    This dashboard analyzes stocks using technical indicators to generate trading signals with confidence scores and recommended timeframes.
    """)
    
    with st.expander("📋 System Overview", expanded=True):
        st.markdown("""
        ### System Architecture
        
        Our technical analysis system uses a hybrid data approach:
        
        1. **Historical Data**: From Alpha Vantage API (free tier, 5 calls/minute)
        2. **Real-time Data**: From Finnhub API (websocket for live trades)
        3. **Technical Analysis**: 8 key indicators to generate signals
        4. **Signal Confidence**: Weighted scoring system (50-95%)
        5. **Timeframe Analysis**: Categorizes optimal holding periods
        6. **Paper Trading**: Virtual portfolio to test signals without real money
        7. **Price Verification**: Compares historical closing prices with real-time data
        
        The system requires a minimum 65% confidence score to display trading signals.
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
    
    with st.expander("📊 Core Indicators"):
        st.markdown("""
        ### Primary Indicators
        
        #### MACD (Moving Average Convergence Divergence)
        
        **What it is**: Trend-following momentum indicator showing relationship between 12-day and 26-day EMAs
        
        **Signal strength**: 15 points for crossover signals
        
        **Optimal timeframe**: Medium-term (2-4 weeks)
        
        ---
        
        #### RSI (Relative Strength Index)
        
        **What it is**: Momentum oscillator measuring overbought/oversold conditions
        
        **Signal strength**: 10 points for overbought/oversold
        
        **Optimal timeframe**: Short-term (3-5 days)
        
        ---
        
        #### Bollinger Bands
        
        **What it is**: Volatility bands around price movement
        
        **Signal strength**: 10 points for price near bands
        
        **Optimal timeframe**: Short-term (3-7 days)
        
        ---
        
        #### EMA Cloud (8 & 21)
        
        **What it is**: Trend identification using short-term EMAs
        
        **Signal strength**: 10 points for alignment
        
        **Optimal timeframe**: Short-term (1-2 weeks)
        
        ---
        
        #### Long-term Trend (EMA50 vs EMA200)
        
        **What it is**: Long-term trend alignment
        
        **Signal strength**: 10 points
        
        **Optimal timeframe**: Long-term (1-3 months)
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
        
        You can execute trades manually from the dashboard or scanner, or set it to automatically execute new signals.
        """)
        
    with st.expander("🔄 Real-time Price Integration"):
        st.markdown("""
        ### Hybrid Data Approach
        
        The system uses a combination of data sources to maximize accuracy:
        
        - **Alpha Vantage**: Used for historical price data and indicator calculations
        - **Finnhub**: Used for real-time price updates and trade execution pricing
        
        When generating signals, the system:
        
        1. Calculates indicators using historical data
        2. Updates the most recent price with real-time data
        3. Adjusts the signal confidence if there's a significant price difference
        4. Flags signals where current and historical prices differ significantly
        
        This approach ensures signals are based on accurate technical analysis while using the most current prices for trade execution.
        """)
    
    with st.expander("📘 API Usage Information"):
        st.markdown("""
        ### API Limitations & Best Practices
        
        #### Alpha Vantage (Historical Data)
        - Free tier limit: 5 API calls per minute, 500 per day
        - Used for: Historical price data, daily OHLC data
        - Premium tier available for higher limits
        
        #### Finnhub (Real-time Data)
        - Free tier includes: Websocket access for real-time trades
        - Used for: Real-time trade monitoring, current quotes
        - Premium tier required for historical OHLCV data
        
        #### Best Practices
        - Limit scanning to 5-10 stocks at a time
        - Allow 60 seconds between scans to respect API limits
        - Use the dashboard for detailed analysis of individual stocks
        - Use scanner sparingly to find potential opportunities
        """)
    
    with st.expander("🏆 Best Practices for Trading"):
        st.markdown("""
        ### How to Use This System Effectively
        
        #### Best Practices
        
        1. **Timeframe Alignment**: Trade according to the recommended holding period
        
        2. **Confirmation**: Look for multiple indicators supporting the same direction
        
        3. **Strong Trends**: Pay attention to ADX values above 25 (indicates strong trend)
        
        4. **Volume Confirmation**: Ensure signals are supported by above-average volume
        
        5. **Risk Management**: Always use stop-losses (typically 5-7% for swing trades)
        
        6. **Price Verification**: Check that real-time prices confirm historical data analysis
        
        #### Risk Management Guidelines
        
        - Position sizing: 1-2% account risk per trade
        - Stop loss placement: Below recent support for buys, above recent resistance for sells
        - Take profits: Use the recommended timeframe as a guide for exit strategy
        """)

if __name__ == "__main__":
    main()
