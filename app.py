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
ALPHA_VANTAGE_API_KEY = "0I7EULOLI2DW2UPW"  # For historical data

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

# Scan for signals across multiple stocks
def scan_for_signals(stocks, min_confidence=65, timeframe="short"):
    all_signals = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(stocks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
            
            # Add a small delay to avoid hitting API rate limits
            if i > 0 and i % 5 == 0:  # Alpha Vantage limit is 5 calls per minute for free tier
                time.sleep(12)  # Wait 12 seconds between every 5 calls
            
            df = fetch_alpha_vantage_daily(symbol, timeframe)
            if df.empty:
                continue
                
            df = calculate_indicators(df)
            signal = generate_signals(df, timeframe)
            
            if signal and signal['confidence'] >= min_confidence and signal['direction'] != 'NEUTRAL':
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                signal['last_price'] = df['close'].iloc[-1]
                all_signals.append(signal)
        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            continue
    
    # Sort by confidence score
    all_signals.sort(key=lambda x: x['confidence'], reverse=True)
    return all_signals

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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Scanner", "Real-time Monitor", "Documentation"])
    
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
    
    # Last price info
    quote = get_current_quote(symbol)
    if quote and 'c' in quote:
        st.sidebar.success(f"Last Price: ${quote['c']:.2f}")
        if 'pc' in quote:
            change = quote['c'] - quote['pc']
            change_pct = (change / quote['pc']) * 100
            color = "green" if change >= 0 else "red"
            st.sidebar.markdown(f"<p style='color:{color}'>Change: {change:.2f} ({change_pct:.2f}%)</p>", unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get data and calculate indicators
        with st.spinner("Fetching data..."):
            data = fetch_alpha_vantage_daily(symbol, selected_timeframe)
        
        if not data.empty:
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
            signal = generate_signals(data_with_indicators, selected_timeframe)
            
            if signal and 'direction' in signal:
                # Display signal with appropriate styling
                if signal['direction'] == 'BUY':
                    st.markdown(f"<p class='buy-signal'>🔵 BUY SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                elif signal['direction'] == 'SELL':
                    st.markdown(f"<p class='sell-signal'>🔴 SELL SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='neutral-signal'>⚪ NEUTRAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                
                # Display timeframe recommendation
                if 'timeframe' in signal:
                    st.markdown(f"<div class='timeframe-info'>✅ <b>Recommended holding period:</b> {signal['timeframe']}</div>", unsafe_allow_html=True)
                
                # Display reasons
                st.subheader("Signal Reasons:")
                for reason in signal['reasons']:
                    st.write(f"• {reason}")
                
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
                signals = scan_for_signals(stocks_to_scan, min_confidence, selected_timeframe)
                
                # Filter by direction if needed
                if signal_direction == "Buy Only":
                    signals = [s for s in signals if s['direction'] == 'BUY']
                elif signal_direction == "Sell Only":
                    signals = [s for s in signals if s['direction'] == 'SELL']
                
                if signals:
                    st.success(f"Found {len(signals)} signals!")
                    
                    # Display signals in a table first (summary)
                    signal_data = []
                    for signal in signals:
                        timeframe_str = signal.get('timeframe', 'Medium-term')
                        signal_data.append({
                            'Symbol': signal['symbol'],
                            'Direction': signal['direction'],
                            'Confidence': f"{signal['confidence']}%",
                            'Last Price': f"${signal['last_price']:.2f}" if 'last_price' in signal else "N/A",
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
                        
                        # Display recommended timeframe
                        if 'timeframe' in signal:
                            st.markdown(f"<div class='timeframe-info'>✅ <b>Recommended holding period:</b> {signal['timeframe']}</div>", unsafe_allow_html=True)
                        
                        # Display current price if available
                        if 'last_price' in signal:
                            st.write(f"Current Price: ${signal['last_price']:.2f}")
                        
                        # Display reasons
                        for reason in signal['reasons']:
                            st.write(f"• {reason}")
                        
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
    
    # Symbol selection
    monitoring_symbols = st.sidebar.multiselect(
        "Select symbols to monitor",
        options=get_us_stock_universe(),
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
                if isinstance(val, str) and '%' in val:
                    val_num = float(val.strip('%'))
                    color = 'green' if val_num >= 0 else 'red'
                elif isinstance(val, (int, float)):
                    color = 'green' if val >= 0 else 'red'
                else:
                    return ''
                return f'color: {color}'
            except:
                return ''
        
        # Apply styling and display
        st.dataframe(df_quotes.style.applymap(color_change, subset=['Change', 'Change %']), use_container_width=True)
    
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
    
    with st.expander("🔢 Signal Generation System"):
        st.markdown("""
        ### Confidence Score Calculation
        
        The system uses a point-based algorithm:
        
        **Starting Base**: 50 points
        
        **Signal Strength**:
        - MACD Crossover: +15 points
        - EMA alignments: +10 points each
        - RSI conditions: +10 points
        - Bollinger Band signals: +10 points
        - Volume confirmation: +10 points
        - ADX trend strength: +10 points
        - Stochastic RSI signals: +5 points
        
        **Final Score**: Capped at 95% confidence
        
        ### Trading Signal Classification
        
        - **BUY Signals**: Generated when bullish indicators outweigh bearish ones
        - **SELL Signals**: Generated when bearish indicators outweigh bullish ones
        - **NEUTRAL**: When there's no clear direction or confidence is below threshold
        
        The recommended timeframe is determined by the strongest contributing signals.
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
        
        #### Risk Management Guidelines
        
        - Position sizing: 1-2% account risk per trade
        - Stop loss placement: Below recent support for buys, above recent resistance for sells
        - Take profits: Use the recommended timeframe as a guide for exit strategy
        """)

if __name__ == "__main__":
    main()
