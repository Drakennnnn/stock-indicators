import streamlit as st
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Technical Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Alpaca API credentials
API_KEY = "PKTR490WK796PV2BCEZ3"
API_SECRET = "iMX9xTr66T0Fra84bTFpglrNkcZh5hATGQjEIsoa"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Cache function for data fetching to avoid repeated API calls
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def fetch_stock_data(symbol, timeframe, limit=100):
    try:
        # Convert timeframe for Alpaca API
        if timeframe == '1min':
            timeframe_api = '1Min'
        elif timeframe == '5min':
            timeframe_api = '5Min'
        elif timeframe == '15min':
            timeframe_api = '15Min'
        elif timeframe == '1hour':
            timeframe_api = '1Hour'
        else:  # '1day'
            timeframe_api = '1Day'
        
        # Fetch data from Alpaca
        bars = api.get_bars(symbol, timeframe_api, limit=limit).df
        
        # Reset index to make date a column
        bars = bars.reset_index()
        
        # For live data, add current time if not already in the data
        if bars.empty or (datetime.now() - bars['timestamp'].iloc[-1]).total_seconds() > 300:
            # No recent data, might be outside market hours or a holiday
            st.info(f"No recent data available for {symbol}. This could be due to market hours or a holiday.")
        
        return bars
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Custom indicator calculation functions
def calculate_indicators(df):
    if len(df) < 50:  # Ensure we have enough data
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
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
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
        df['volume_z_score'] = (df['volume'] - df['volume_sma20']) / df['volume_std']
        
        # On Balance Volume (OBV)
        obv = pd.Series(index=df.index)
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
    
    # ADX calculation is more complex, so we'll use a simplified version
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
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=smooth_period).mean() / df['tr'].rolling(window=smooth_period).mean()
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=smooth_period).mean() / df['tr'].rolling(window=smooth_period).mean()
        
        # Calculate DX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=smooth_period).mean()
    except Exception as e:
        st.warning(f"Error calculating ADX: {e}")
    
    return df

# Generate trading signals
def generate_signals(df):
    if df.empty or len(df) < 50:
        return []
    
    signals = []
    confidence_score = 50  # Base confidence score
    
    # Check last row for signals
    last_idx = -1
    
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
        if 'bb_width' in df.columns:
            bb_mean = df['bb_width'].rolling(window=20).mean().iloc[last_idx]
            if df['bb_width'].iloc[last_idx] < bb_mean * 0.8:
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

# Plot chart with indicators
def plot_chart(df, symbol, timeframe):
    if df.empty or len(df) < 20:
        st.error("Not enough data to plot chart")
        return
    
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
    if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
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
    if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_hist' in df.columns:
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
        title=f'{symbol} - {timeframe} Timeframe',
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

# Check if market is open
def is_market_open():
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        st.error(f"Error checking market status: {e}")
        return False

# Get universe of stocks 
def get_stock_universe():
    # Default list for demo purposes
    default_stocks = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "TSLA", "NVDA", "JPM", "BAC", "V", 
        "JNJ", "PG", "UNH", "HD", "XOM"
    ]
    
    try:
        # Get S&P 500 stocks (simplified approach)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        if len(tables) > 0 and 'Symbol' in tables[0].columns:
            sp500_stocks = tables[0]['Symbol'].tolist()
            # Clean up symbols
            sp500_stocks = [symbol.replace('.', '-') for symbol in sp500_stocks]
            return sp500_stocks
        else:
            return default_stocks
    except Exception as e:
        st.warning(f"Unable to fetch S&P 500 list, using default stocks: {e}")
        return default_stocks

# Scan for signals across multiple stocks
def scan_for_signals(stocks, timeframe, min_confidence=65):
    all_signals = []
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(stocks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
            
            df = fetch_stock_data(symbol, timeframe)
            if df.empty:
                continue
                
            df = calculate_indicators(df)
            signal = generate_signals(df)
            
            if signal and signal['confidence'] >= min_confidence and signal['direction'] != 'NEUTRAL':
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_signals.append(signal)
        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")
            continue
    
    # Sort by confidence score
    all_signals.sort(key=lambda x: x['confidence'], reverse=True)
    return all_signals

# App navigation
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Stock Scanner", "Documentation"])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Scanner":
        show_scanner()
    elif page == "Documentation":
        show_documentation()

# Dashboard page
def show_dashboard():
    st.title("📈 Technical Trading Signal Dashboard")
    
    # Settings in sidebar
    st.sidebar.header("Dashboard Settings")
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
    timeframe = st.sidebar.selectbox(
        "Timeframe", 
        ["1min", "5min", "15min", "1hour", "1day"],
        index=3  # Default to 1hour
    )
    
    # Auto refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    
    # Market status
    market_open = is_market_open()
    
    if market_open:
        st.sidebar.success("✅ Market is OPEN")
    else:
        st.sidebar.warning("⚠️ Market is CLOSED")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get data and calculate indicators
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(symbol, timeframe)
        
        if not data.empty:
            with st.spinner("Calculating indicators..."):
                data_with_indicators = calculate_indicators(data)
            
            # Plot chart
            fig = plot_chart(data_with_indicators, symbol, timeframe)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No data available for {symbol} with {timeframe} timeframe")
    
    with col2:
        st.subheader("Signal Analysis")
        
        if not data.empty:
            signal = generate_signals(data_with_indicators)
            
            if signal and 'direction' in signal:
                # Display signal with appropriate styling
                if signal['direction'] == 'BUY':
                    st.success(f"🔵 BUY SIGNAL - {signal['confidence']}% Confidence")
                elif signal['direction'] == 'SELL':
                    st.error(f"🔴 SELL SIGNAL - {signal['confidence']}% Confidence")
                else:
                    st.info(f"⚪ NEUTRAL - {signal['confidence']}% Confidence")
                
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
                    st.write(f"RSI: {data_with_indicators['rsi'].iloc[-1]:.2f}")
                if 'macd' in data_with_indicators.columns:
                    st.write(f"MACD: {data_with_indicators['macd'].iloc[-1]:.2f}")
                if 'macd_signal' in data_with_indicators.columns:
                    st.write(f"MACD Signal: {data_with_indicators['macd_signal'].iloc[-1]:.2f}")
                if 'adx' in data_with_indicators.columns:
                    st.write(f"ADX: {data_with_indicators['adx'].iloc[-1]:.2f}")
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
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe", 
        ["5min", "15min", "1hour", "1day"],
        index=2  # Default to 1hour
    )
    
    # Minimum confidence threshold
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 
        50, 95, 65
    )
    
    # Stocks to scan
    stocks_to_scan = []
    
    if scan_mode == "Quick Scan (Pre-selected)":
        stocks_to_scan = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "JPM", "BAC", "V", 
            "JNJ", "PG", "UNH", "HD", "XOM"
        ]
        st.write(f"Scanning top 15 stocks at {timeframe} timeframe...")
    else:
        # Custom stock input
        custom_input = st.sidebar.text_area(
            "Enter stock symbols (comma-separated)",
            "AAPL, MSFT, AMZN, GOOGL, META"
        )
        
        # Parse input
        stocks_to_scan = [s.strip().upper() for s in custom_input.split(",")]
        st.write(f"Scanning {len(stocks_to_scan)} custom stocks at {timeframe} timeframe...")
    
    # Scan button
    if st.button("Run Scan"):
        with st.spinner("Scanning for signals..."):
            signals = scan_for_signals(stocks_to_scan, timeframe, min_confidence)
            
            if signals:
                st.success(f"Found {len(signals)} signals!")
                
                # Display signals in a nice format
                for i, signal in enumerate(signals, 1):
                    if signal['direction'] == 'BUY':
                        st.markdown(f"""
                        ### 🔵 BUY - {signal['symbol']} - {signal['confidence']}% Confidence
                        """)
                    else:
                        st.markdown(f"""
                        ### 🔴 SELL - {signal['symbol']} - {signal['confidence']}% Confidence
                        """)
                    
                    # Display reasons
                    for reason in signal['reasons']:
                        st.write(f"• {reason}")
                    
                    # Add a separator between signals
                    if i < len(signals):
                        st.markdown("---")
            else:
                st.info("No strong signals found matching your criteria")

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
        
        1. **Data Acquisition**: Fetches live market data from Alpaca API
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
        
        #### Moving Average Trend Alignment
        
        **What it is**: Uses the relationship between longer-term moving averages to confirm trend direction.
        
        **How it's calculated**: 
        - EMA50 vs EMA200 position
        
        **How it's used**:
        - EMA50 > EMA200: Long-term uptrend (bullish)
        - EMA50 < EMA200: Long-term downtrend (bearish)
        
        **Points awarded**: Integrated into overall trend confirmation
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
        - Multi-timeframe confirmation: +10 points
        
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
        
        2. **Trending Markets**: Technical indicators work best in trending markets (ADX > 20)
        
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
        
        1. **Delayed Data**: Free API tiers may have slightly delayed data
        
        2. **Processing Time**: Scanning many stocks simultaneously takes time
        
        3. **Historical Context**: System uses limited historical data for calculations
        
        #### Remember
        
        This system is a tool to identify potential trades with higher probability, not a guaranteed profit system. Always combine with fundamental analysis and proper risk management for best results.
        """)

# Run the app
if __name__ == "__main__":
    main()
