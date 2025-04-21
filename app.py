import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import finnhub
import time
from datetime import datetime, timedelta
import pytz
import websocket
import json
import threading
import queue

# Set page configuration
st.set_page_config(
    page_title="Technical Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Finnhub API credentials
FINNHUB_API_KEY = "d03bkkpr01qvvb93ems0d03bkkpr01qvvb93emsg"

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Message queue for websocket data
ws_message_queue = queue.Queue()

# Initialize websocket connection
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

# Cache function for data fetching to avoid repeated API calls
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def fetch_stock_data(symbol, resolution, days_back=30):
    try:
        # Calculate time range
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        # Fetch candle data
        candles = finnhub_client.stock_candles(
            symbol, 
            resolution, 
            start_time, 
            end_time
        )
        
        if candles['s'] == 'ok':
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime([datetime.fromtimestamp(t) for t in candles['t']]),
                'open': candles['o'],
                'high': candles['h'],
                'low': candles['l'],
                'close': candles['c'],
                'volume': candles['v']
            })
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Calculate all technical indicators
def calculate_indicators(df):
    if len(df) < 30:  # Ensure we have enough data
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
def generate_signals(df):
    if df.empty or len(df) < 30:
        return None
    
    signals = []
    confidence_score = 50  # Base confidence score
    
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
                    'points': 15
                })
                confidence_score += 15
            
            # Bearish MACD crossover
            elif df['macd_cross_down'].iloc[last_idx]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'MACD Bearish Crossover',
                    'points': 15
                })
                confidence_score += 15
        
        # EMA Cloud
        if 'ema_cloud_bullish' in df.columns:
            if df['ema_cloud_bullish'].iloc[last_idx]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'EMA Cloud Bullish (EMA8 > EMA21)',
                    'points': 10
                })
                confidence_score += 10
            else:
                signals.append({
                    'type': 'SELL',
                    'reason': 'EMA Cloud Bearish (EMA8 < EMA21)',
                    'points': 10
                })
                confidence_score += 10
        
        # RSI Signals
        if 'rsi' in df.columns:
            # Oversold
            if df['rsi'].iloc[last_idx] < 30:
                signals.append({
                    'type': 'BUY',
                    'reason': 'RSI Oversold (<30)',
                    'points': 10
                })
                confidence_score += 10
            # Overbought
            elif df['rsi'].iloc[last_idx] > 70:
                signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought (>70)',
                    'points': 10
                })
                confidence_score += 10
        
        # Stochastic RSI Signals
        if 'stoch_rsi_k' in df.columns and 'stoch_rsi_d' in df.columns:
            # Oversold
            if df['stoch_rsi_k'].iloc[last_idx] < 20:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Stochastic RSI Oversold (<20)',
                    'points': 5
                })
                confidence_score += 5
            # Overbought
            elif df['stoch_rsi_k'].iloc[last_idx] > 80:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Stochastic RSI Overbought (>80)',
                    'points': 5
                })
                confidence_score += 5
            
            # Crossover
            if df['stoch_rsi_k'].iloc[last_idx] > df['stoch_rsi_d'].iloc[last_idx] and df['stoch_rsi_k'].iloc[last_idx-1] <= df['stoch_rsi_d'].iloc[last_idx-1]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Stochastic RSI Bullish Crossover',
                    'points': 5
                })
                confidence_score += 5
            elif df['stoch_rsi_k'].iloc[last_idx] < df['stoch_rsi_d'].iloc[last_idx] and df['stoch_rsi_k'].iloc[last_idx-1] >= df['stoch_rsi_d'].iloc[last_idx-1]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Stochastic RSI Bearish Crossover',
                    'points': 5
                })
                confidence_score += 5
        
        # Bollinger Band signals
        if 'bb_pct_b' in df.columns and 'bb_width' in df.columns:
            # Price near upper band
            if df['bb_pct_b'].iloc[last_idx] > 0.9:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Price near upper Bollinger Band',
                    'points': 10
                })
                confidence_score += 10
            # Price near lower band
            elif df['bb_pct_b'].iloc[last_idx] < 0.1:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Price near lower Bollinger Band',
                    'points': 10
                })
                confidence_score += 10
            
            # Bollinger Band squeeze (setup for volatility breakout)
            if df['bb_width'].iloc[last_idx] < df['bb_width'].rolling(window=20).mean().iloc[last_idx] * 0.8:
                signals.append({
                    'type': 'NEUTRAL',
                    'reason': 'Bollinger Band Squeeze (potential breakout setup)',
                    'points': 5
                })
                confidence_score += 5
        
        # Volume confirmation
        if 'volume_z_score' in df.columns:
            if df['volume_z_score'].iloc[last_idx] > 1.5:
                # High volume confirms the direction
                direction = 'BUY' if df['close'].iloc[last_idx] > df['close'].iloc[last_idx-1] else 'SELL'
                signals.append({
                    'type': direction,
                    'reason': 'High Volume Confirmation',
                    'points': 10
                })
                confidence_score += 10
        
        # ADX - Strong trend
        if 'adx' in df.columns:
            if df['adx'].iloc[last_idx] > 25:
                signals.append({
                    'type': 'NEUTRAL',
                    'reason': 'Strong Trend (ADX > 25)',
                    'points': 10
                })
                confidence_score += 10
        
        # Determine overall signal direction and confidence
        buy_points = sum(signal['points'] for signal in signals if signal['type'] == 'BUY')
        sell_points = sum(signal['points'] for signal in signals if signal['type'] == 'SELL')
        
        # Calculate final confidence (cap at 95)
        final_confidence = min(95, confidence_score)
        
        # Determine overall signal
        if buy_points > sell_points and final_confidence >= 65:
            overall_signal = {
                'direction': 'BUY',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'BUY' or s['type'] == 'NEUTRAL']
            }
        elif sell_points > buy_points and final_confidence >= 65:
            overall_signal = {
                'direction': 'SELL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'SELL' or s['type'] == 'NEUTRAL']
            }
        else:
            overall_signal = {
                'direction': 'NEUTRAL',
                'confidence': final_confidence,
                'signals': signals,
                'reasons': [s['reason'] for s in signals if s['type'] == 'NEUTRAL']
            }
        
        return overall_signal
    
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return None

# Plot chart with indicators
def plot_chart(df, symbol, resolution):
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
            title=f'{symbol} - {resolution} Resolution',
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

# Get available symbols
def get_stock_universe():
    # Default list of US stocks
    default_stocks = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "TSLA", "NVDA", "JPM", "BAC", "V", 
        "JNJ", "PG", "UNH", "HD", "XOM"
    ]
    
    # Default list of Indian stocks
    india_stocks = [
        "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
        "HINDUNILVR.BO", "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "KOTAKBANK.BO"
    ]
    
    return default_stocks + india_stocks

# Scan for signals across multiple stocks
def scan_for_signals(stocks, resolution, min_confidence=65):
    all_signals = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(stocks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
            
            # Add a small delay to avoid hitting API rate limits (30 calls/sec for Finnhub)
            if i > 0 and i % 25 == 0:
                time.sleep(1)
            
            df = fetch_stock_data(symbol, resolution)
            if df.empty:
                continue
                
            df = calculate_indicators(df)
            signal = generate_signals(df)
            
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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Scanner", "Real-time Monitor", "Documentation"])
    
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
    
    # Region selection
    region = st.sidebar.selectbox(
        "Market Region", 
        ["US", "India"],
        index=0
    )
    
    # Stock selection based on region
    if region == "US":
        default_symbol = "AAPL"
        symbol_suffix = ""
    else:  # India
        default_symbol = "RELIANCE"
        symbol_suffix = ".BO"
    
    symbol = st.sidebar.text_input("Stock Symbol", default_symbol).upper()
    
    # Add suffix for Indian stocks if not already present
    if region == "India" and not symbol.endswith(".BO"):
        symbol = symbol + symbol_suffix
    
    # Resolution selection
    resolution = st.sidebar.selectbox(
        "Resolution", 
        ["D", "60", "30", "15", "5", "1"],
        index=0  # Default to Daily
    )
    
    # Convert resolution to days for lookback
    if resolution == "D":
        days_back = 200  # For longer-term view on daily resolution
    elif resolution == "60":
        days_back = 30
    else:
        days_back = 14
    
    # Auto refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    
    # Last price info
    try:
        last_price = finnhub_client.quote(symbol)
        if 'c' in last_price:
            st.sidebar.success(f"Last Price: ${last_price['c']:.2f}")
            if 'pc' in last_price:
                change = last_price['c'] - last_price['pc']
                change_pct = (change / last_price['pc']) * 100
                color = "green" if change >= 0 else "red"
                st.sidebar.markdown(f"<p style='color:{color}'>Change: {change:.2f} ({change_pct:.2f}%)</p>", unsafe_allow_html=True)
    except:
        st.sidebar.warning("Unable to fetch latest price")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get data and calculate indicators
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(symbol, resolution, days_back)
        
        if not data.empty:
            with st.spinner("Calculating indicators..."):
                data_with_indicators = calculate_indicators(data)
            
            # Plot chart
            fig = plot_chart(data_with_indicators, symbol, resolution)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No data available for {symbol} with {resolution} resolution")
    
    with col2:
        st.subheader("Signal Analysis")
        
        if not data.empty:
            signal = generate_signals(data_with_indicators)
            
            if signal and 'direction' in signal:
                # Display signal with appropriate styling
                if signal['direction'] == 'BUY':
                    st.markdown(f"<p class='buy-signal'>🔵 BUY SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                elif signal['direction'] == 'SELL':
                    st.markdown(f"<p class='sell-signal'>🔴 SELL SIGNAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='neutral-signal'>⚪ NEUTRAL - {signal['confidence']}% Confidence</p>", unsafe_allow_html=True)
                
                # Display reasons
                st.subheader("Signal Reasons:")
                for reason in signal['reasons']:
                    st.write(f"• {reason}")
                
                # Display latest price data
                latest = data.iloc[-1]
                st.subheader("Latest Price Data:")
                st.write(f"Open: ${latest['open']:.2f}")
                st.write(f"High: ${latest['high']:.2f}")
                st.write(f"Low: ${latest['low']:.2f}")
                st.write(f"Close: ${latest['close']:.2f}")
                st.write(f"Volume: {int(latest['volume']):,}")
                st.write(f"Time: {latest['timestamp']}")
                
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
        st.write(f"Page will refresh in {refresh_interval} seconds")
        time.sleep(refresh_interval)
        st.experimental_rerun()

# Scanner page
def show_scanner():
    st.title("🔍 Stock Signal Scanner")
    
    # Settings
    st.sidebar.header("Scanner Settings")
    
    # Choose scan mode
    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (Pre-selected)", "Custom Scan"]
    )
    
    # Region selection
    region = st.sidebar.selectbox(
        "Market Region", 
        ["US", "India", "Both"],
        index=0
    )
    
    # Resolution selection
    resolution = st.sidebar.selectbox(
        "Resolution", 
        ["D", "60", "30", "15", "5"],
        index=0  # Default to Daily
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
    
    if scan_mode == "Quick Scan (Pre-selected)":
        if region == "US" or region == "Both":
            stocks_to_scan += [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
                "TSLA", "NVDA", "JPM", "BAC", "V", 
                "JNJ", "PG", "UNH", "HD", "XOM"
            ]
        
        if region == "India" or region == "Both":
            stocks_to_scan += [
                "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
                "HINDUNILVR.BO", "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "KOTAKBANK.BO"
            ]
        
        st.write(f"Scanning {len(stocks_to_scan)} stocks at {resolution} resolution...")
    else:
        # Custom stock input
        custom_input = st.sidebar.text_area(
            "Enter stock symbols (comma-separated)",
            "AAPL, MSFT, AMZN, GOOGL, META"
        )
        
        # Parse input
        stocks_to_scan = [s.strip().upper() for s in custom_input.split(",")]
        
        # Add BO suffix to Indian stocks if region is India and they don't have it
        if region == "India":
            stocks_to_scan = [s + ".BO" if not s.endswith(".BO") else s for s in stocks_to_scan]
        
        st.write(f"Scanning {len(stocks_to_scan)} custom stocks at {resolution} resolution...")
    
    # Display stock list
    with st.expander("View stocks to scan"):
        st.write(", ".join(stocks_to_scan))
    
    # Scan button
    if st.button("Run Scan"):
        with st.spinner("Scanning for signals..."):
            signals = scan_for_signals(stocks_to_scan, resolution, min_confidence)
            
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
                    signal_data.append({
                        'Symbol': signal['symbol'],
                        'Direction': signal['direction'],
                        'Confidence': f"{signal['confidence']}%",
                        'Last Price': f"${signal['last_price']:.2f}" if 'last_price' in signal else "N/A",
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
                st.info("No strong signals found matching your criteria")

# Real-time Monitor
def show_real_time_monitor():
    st.title("📊 Real-time Market Monitor")
    
    # Settings
    st.sidebar.header("Real-time Settings")
    
    # Symbol selection
    monitoring_symbols = st.sidebar.multiselect(
        "Select symbols to monitor",
        options=get_stock_universe(),
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
            # Add a small delay to avoid API rate limits
            time.sleep(0.1)
            
            quote = finnhub_client.quote(symbol)
            if 'c' in quote:
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
        st.experimental_rerun()

# Documentation page
def show_documentation():
    st.title("📚 Technical Indicator Documentation")
    
    st.markdown("""
    This documentation explains the technical indicators used in this system and how they're applied to generate trading signals.
    """)
    
    with st.expander("System Overview", expanded=True):
        st.markdown("""
        ### System Architecture
        
        Our technical analysis system follows this process flow:
        
        1. **Data Acquisition**: Fetches market data from Finnhub API
        2. **Indicator Calculation**: Calculates technical indicators on the data
        3. **Signal Generation**: Identifies potential trading signals
        4. **Confidence Scoring**: Assigns confidence scores to signals
        5. **Trade Selection**: Filters for high-probability trades
        
        The system uses a point-based scoring system, with trades requiring a minimum 65% confidence score to be displayed.
        """)
    
    with st.expander("Core Indicators"):
        st.markdown("""
        ### Primary Indicators
        
        #### MACD (Moving Average Convergence Divergence)
        
        **What it is**: The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
        
        **How it's calculated**: 
        - MACD Line = 12-period EMA - 26-period EMA
        - Signal Line = 9-period EMA of MACD Line
        - Histogram = MACD Line - Signal Line
        
        **How it's used**:
        - Bullish signal: MACD line crosses above signal line
        - Bearish signal: MACD line crosses below signal line
        - Histogram increasing: Momentum is increasing
        - Zero line crossovers: Potential trend changes
        
        **Points awarded**: 15 points for crossover signals
        
        ---
        
        #### RSI (Relative Strength Index)
        
        **What it is**: A momentum oscillator that measures the speed and change of price movements, oscillating between 0 and 100.
        
        **How it's calculated**: 
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over 14 periods
        
        **How it's used**:
        - Overbought: RSI > 70
        - Oversold: RSI < 30
        - Divergence: Price makes new high/low but RSI doesn't confirm
        
        **Points awarded**: 10 points for oversold/overbought conditions
        
        ---
        
        #### Bollinger Bands
        
        **What it is**: A volatility indicator that creates a band around the price movement.
        
        **How it's calculated**: 
        - Middle Band = 20-period SMA
        - Upper Band = Middle Band + (2 * Standard Deviation)
        - Lower Band = Middle Band - (2 * Standard Deviation)
        - %B = (Price - Lower Band) / (Upper Band - Lower Band)
        
        **How it's used**:
        - Price near upper band: Potentially overbought
        - Price near lower band: Potentially oversold
        - Narrow bands (squeeze): Potential for volatility breakout
        - Wide bands: High volatility environment
        
        **Points awarded**: 10 points for price near bands, 5 points for squeeze
        
        ---
        
        #### EMA Cloud (8 & 21)
        
        **What it is**: A trend identification system using two exponential moving averages.
        
        **How it's calculated**: 
        - Fast EMA = 8-period EMA
        - Slow EMA = 21-period EMA
        
        **How it's used**:
        - Bullish: Fast EMA above Slow EMA
        - Bearish: Fast EMA below Slow EMA
        - Crossovers: Potential trend changes
        
        **Points awarded**: 10 points for alignment with direction
        """)
    
    with st.expander("Supporting Indicators"):
        st.markdown("""
        ### Secondary Indicators
        
        #### Volume Analysis
        
        **Components**:
        - Volume Z-Score: Measures how current volume compares to average
        - On Balance Volume (OBV): Cumulative indicator that adds volume on up days and subtracts on down days
        
        **How it's used**:
        - High volume confirms price movement
        - Divergence between OBV and price can signal potential reversals
        
        **Points awarded**: 10 points for strong volume confirmation
        
        ---
        
        #### ADX (Average Directional Index)
        
        **What it is**: Measures trend strength regardless of direction.
        
        **How it's calculated**: Uses the difference between +DI and -DI smoothed over 14 periods.
        
        **How it's used**:
        - ADX > 25: Strong trend present
        - ADX < 20: Weak or no trend
        
        **Points awarded**: 10 points for strong trend identification
        
        ---
        
        #### Stochastic RSI
        
        **What it is**: Applies the Stochastic formula to RSI values instead of price.
        
        **How it's calculated**: 
        - Take RSI values and calculate where current RSI stands relative to its high/low range over 14 periods
        
        **How it's used**:
        - Values above 80 are considered overbought
        - Values below 20 are considered oversold
        - Crossovers between %K and %D lines signal potential reversals
        
        **Points awarded**: 5 points for oversold/overbought conditions, 5 points for crossovers
        """)
    
    with st.expander("Signal Generation System"):
        st.markdown("""
        ### Signal Generation Process
        
        #### Confidence Scoring
        
        The system uses a point-based algorithm to determine confidence:
        
        **Starting Base**: 50 points
        
        **Additional Points**:
        - MACD Crossover: +15 points
        - EMA Cloud alignment: +10 points
        - RSI conditions: +10 points
        - Bollinger Band signals: +10 points
        - Volume confirmation: +10 points
        - ADX trend strength: +10 points
        - Stochastic RSI signals: +5 points
        
        The final confidence score is capped at 95%, acknowledging that no trading signal can be 100% certain.
        
        #### Signal Types
        
        The system identifies three types of signals:
        
        1. **BUY Signals**: Generated when bullish indicators outweigh bearish ones
           - Highest confidence buy signals typically have MACD, EMA, and volume all aligned
        
        2. **SELL Signals**: Generated when bearish indicators outweigh bullish ones
           - Highest confidence sell signals typically have MACD crossdown with overbought RSI
        
        3. **NEUTRAL**: When there's no clear direction or confidence is below threshold
        
        #### Filtering Logic
        
        Only signals with 65% or higher confidence are displayed, focusing on high-probability setups.
        """)
    
    with st.expander("Best Practices & Trading Strategy"):
        st.markdown("""
        ### How to Use This System Effectively
        
        #### Best Practices
        
        1. **Confluence of Signals**: The most reliable trades occur when multiple indicators align
        
        2. **Trending Markets**: Technical indicators work best in trending markets (ADX > 25)
        
        3. **Timeframe Alignment**: Check multiple timeframes for confirmation
           - Ideal setup: Signal present on 15-min, 1-hour, and daily charts
        
        4. **Volume Confirmation**: Always verify signals with volume analysis
           - Strong signals should have above-average volume
        
        5. **Risk Management**: No technical system is 100% accurate
           - Always use stop-losses
           - Position sizing based on account risk (1-2% maximum risk per trade)
        
        #### Highest Probability Setups
        
        1. **Momentum + Confirmation + Volume Trio**:
           - RSI moving up from midrange (40-60)
           - MACD bullish crossover
           - Above average volume
        
        2. **Reversal Setups**:
           - RSI divergence (price makes new high/low but RSI doesn't)
           - MACD histogram reversal
           - Volume spike on reversal day
        
        3. **Trend Continuation**:
           - Price pullback to EMA8/21 cloud
           - MACD stays above zero line (for uptrends)
           - Volume decreases on pullback, increases on continuation
        """)
    
    with st.expander("System Limitations"):
        st.markdown("""
        ### Understanding System Limitations
        
        #### Market Condition Limitations
        
        1. **Sideways Markets**: Technical indicators generate more false signals in non-trending markets
        
        2. **Extreme Volatility**: During market crashes or unusual events, correlations and patterns may break down
        
        3. **Low Liquidity**: Signals may be less reliable for stocks with low trading volume
        
        #### Technical Limitations
        
        1. **API Restrictions**: Finnhub's free tier has limitations on data availability and request frequency
        
        2. **Processing Time**: Scanning many stocks simultaneously takes time
        
        3. **Historical Context**: System uses limited historical data for calculations
        
        #### Remember
        
        This system is a tool to identify potential trades with higher probability, not a guaranteed profit system. Always combine with fundamental analysis and proper risk management for best results.
        """)
    
    with st.expander("About Finnhub API"):
        st.markdown("""
        ### About Finnhub API
        
        This application uses Finnhub.io API for market data. Finnhub provides:
        
        - Real-time stock data
        - Technical indicators
        - Company information
        - News and sentiment analysis
        
        The free tier of Finnhub allows:
        - 60 API calls per minute
        - Access to delayed market data
        - Limited historical data
        
        For more information, visit [Finnhub.io](https://finnhub.io/)
        """)

# Run the app
if __name__ == "__main__":
    main()
