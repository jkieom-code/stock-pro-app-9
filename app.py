import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

# --- Configuration ---
st.set_page_config(
    page_title="ProStock | AI-Powered Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa; /* Softer white/gray background */
        color: #212529;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 28px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Metric Cards Custom Container */
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* Analysis Box */
    .ai-analysis-box {
        background-color: #ffffff;
        border-left: 5px solid #0d6efd;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-top: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 10px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e7f1ff;
        color: #0d6efd;
        font-weight: bold;
    }

    /* Fix Layout Overlap */
    .block-container {
        padding-top: 5rem; /* Increased padding to clear top bar */
        max-width: 95%;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=60)
def get_stock_data(ticker, interval, period, start=None, end=None):
    try:
        # 1. Primary Attempt
        if interval == "1d" and start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # 2. Fallback Logic (Crucial for Weekends/Market Holidays)
        # If '1d' period returns empty (because market is closed today), fetch last 5 days
        if data.empty and period == "1d":
            data = yf.download(ticker, period="5d", interval=interval, progress=False)
        
        return data
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news
    except Exception:
        return {}, []

@st.cache_data(ttl=300)
def get_exchange_rate(pair="KRW=X"):
    """Fetches the current exchange rate for USD to KRW."""
    try:
        data = yf.Ticker(pair).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        return None
    return None

def calculate_currency_conversion(amount, from_curr, to_curr):
    """Calculates live currency conversion."""
    if from_curr == to_curr:
        return amount, 1.0
    
    try:
        # Try direct pair first
        ticker = f"{from_curr}{to_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            return amount * rate, rate
            
        # Try inverse
        ticker = f"{to_curr}{from_curr}=X"
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            rate = 1.0 / data['Close'].iloc[-1]
            return amount * rate, rate
    except:
        pass
    
    return None, None

def calculate_technicals(data):
    if len(data) < 2: return data
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # SMA / EMA
    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * std
    data['BB_Lower'] = data['BB_Middle'] - 2 * std
    
    return data

def get_fear_and_greed_proxy():
    """Proxy F&G using VIX and Momentum to avoid scraping blocks."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        sp500 = yf.Ticker("^GSPC").history(period="6mo")
        if sp500.empty: return 50, "Neutral"
        
        current_sp = sp500['Close'].iloc[-1]
        avg_sp = sp500['Close'].mean()
        
        # VIX: Lower is Greed (Bullish), Higher is Fear (Bearish)
        fear_score = max(0, min(100, 100 - (vix - 10) * 2.5))
        
        # Momentum: Higher is Greed
        momentum_score = max(0, min(100, 50 + ((current_sp - avg_sp) / avg_sp) * 500))
        
        final_score = (fear_score * 0.4) + (momentum_score * 0.6)
        
        if final_score < 25: label = "Extreme Fear"
        elif final_score < 45: label = "Fear"
        elif final_score < 55: label = "Neutral"
        elif final_score < 75: label = "Greed"
        else: label = "Extreme Greed"
        
        return int(final_score), label
    except:
        return 50, "Neutral"

def safe_extract_news_title(item):
    """Recursively search for a title in a messy dictionary."""
    if not isinstance(item, dict):
        return None
    
    if 'title' in item and item['title']:
        return item['title']
    
    if 'content' in item and isinstance(item['content'], dict):
        if 'title' in item['content'] and item['content']['title']:
            return item['content']['title']
            
    for key, value in item.items():
        if isinstance(value, dict):
            res = safe_extract_news_title(value)
            if res: return res
            
    return None

def analyze_news_sentiment(news_items):
    if not news_items: return 0, 0, 0, "Neutral"
    
    polarities = []
    for item in news_items:
        title = safe_extract_news_title(item)
        if title:
            blob = TextBlob(title)
            polarities.append(blob.sentiment.polarity)
            
    if not polarities: return 0, 0, 0, "Neutral"
    
    pos = sum(1 for p in polarities if p > 0.05)
    neg = sum(1 for p in polarities if p < -0.05)
    neu = len(polarities) - pos - neg
    
    avg_pol = np.mean(polarities)
    if avg_pol > 0.05: label = "Positive"
    elif avg_pol < -0.05: label = "Negative"
    else: label = "Neutral"
    
    return pos, neg, neu, label

def generate_ai_report(ticker, price, sma, rsi, fg_score, fg_label, news_label):
    report = f"### 🧠 AI Executive Summary for {ticker}\n\n"
    
    # Sentiment Section
    report += f"**1. Market Sentiment:**\n"
    report += f"The market is currently driven by **{fg_label} ({fg_score}/100)**. "
    if fg_score < 40: report += "High fear levels suggest oversold conditions. Contrarian buy signals may be forming.\n"
    elif fg_score > 60: report += "High greed levels suggest overbought conditions. Risk of correction is elevated.\n"
    else: report += "Sentiment is balanced. Market is looking for a catalyst.\n"
    
    # News Section
    report += f"\n**2. News Analysis:**\n"
    report += f"Headlines are trending **{news_label}**. "
    if news_label == "Positive": report += "Optimistic coverage is providing tailwinds for the asset.\n"
    elif news_label == "Negative": report += "Pessimistic coverage is creating headwinds. Watch for volatility.\n"
    else: report += "Coverage is mixed or neutral, implying no major narrative shift.\n"
    
    # Technical Section
    trend = "Bullish 🟢" if price > sma else "Bearish 🔴"
    rsi_state = "Overbought ⚠️" if rsi > 70 else "Oversold 🛒" if rsi < 30 else "Neutral ⚖️"
    
    report += f"\n**3. Technical Outlook:**\n"
    report += f"- **Trend:** {trend} (Price vs {sma_period}-period SMA)\n"
    report += f"- **Momentum:** {rsi_state} (RSI: {rsi:.1f})\n"
    
    return report

# --- Sidebar ---
st.sidebar.markdown("## 📈 ProStock Terminal")
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.rerun()

st.sidebar.markdown("---")

# --- Asset Class Selection ---
market_type = st.sidebar.selectbox(
    "Market Type",
    ["Stocks", "Commodities", "Currencies/Forex"],
    index=0
)

ticker = "" # Initialize

if market_type == "Stocks":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
elif market_type == "Commodities":
    # Dictionary of common commodities and their futures tickers
    commodities = {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Copper": "HG=F",
        "Natural Gas": "NG=F",
        "Corn": "ZC=F",
        "Soybeans": "ZS=F"
    }
    selected_comm = st.sidebar.selectbox("Select Commodity", list(commodities.keys()))
    ticker = commodities[selected_comm]
elif market_type == "Currencies/Forex":
    # Dictionary of common forex pairs
    currencies = {
        "USD/KRW (Won)": "KRW=X",
        "EUR/USD": "EURUSD=X",
        "JPY/USD": "JPY=X",
        "GBP/USD": "GBPUSD=X",
        "Bitcoin (USD)": "BTC-USD",
        "Ethereum (USD)": "ETH-USD"
    }
    selected_curr = st.sidebar.selectbox("Select Pair", list(currencies.keys()))
    ticker = currencies[selected_curr]

# Timeframe Selector
st.sidebar.markdown("### ⏱️ Timeframe")
timeframe = st.sidebar.selectbox(
    "Select Interval",
    ["1 Minute", "5 Minute", "1 Hour", "1 Day"],
    index=0,
    label_visibility="collapsed"
)

# Logic for interval/period
if timeframe == "1 Minute":
    interval = "1m"
    period = "1d"
elif timeframe == "5 Minute":
    interval = "5m"
    period = "5d"
elif timeframe == "1 Hour":
    interval = "1h"
    period = "1mo"
else:
    interval = "1d"
    period = "1y"

# Date range only for Daily
if interval == "1d":
    start_date = st.sidebar.date_input("Start", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End", value=datetime.now())
else:
    st.sidebar.caption(f"Live Feed: Last {period}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Indicators")
show_sma = st.sidebar.toggle("SMA (Simple MA)", value=True)
sma_period = st.sidebar.number_input("SMA Period", value=20) if show_sma else 20
show_ema = st.sidebar.toggle("EMA (Exponential MA)")
ema_period = st.sidebar.number_input("EMA Period", value=50) if show_ema else 50
show_bb = st.sidebar.toggle("Bollinger Bands")
show_rsi = st.sidebar.toggle("RSI (Momentum)")

st.sidebar.markdown("---")

# --- Currency Converter Tool ---
with st.sidebar.expander("🧮 Currency Converter", expanded=False):
    cc_amount = st.number_input("Amount", value=100.0, min_value=0.0)
    c1, c2 = st.columns(2)
    with c1:
        cc_from = st.selectbox("From", ["USD", "KRW", "EUR", "JPY", "GBP", "CNY", "BTC"], index=0)
    with c2:
        cc_to = st.selectbox("To", ["KRW", "USD", "EUR", "JPY", "GBP", "CNY", "BTC"], index=0)
    
    if st.button("Convert"):
        res, rate = calculate_currency_conversion(cc_amount, cc_from, cc_to)
        if res is not None:
            st.success(f"{cc_amount:,.2f} {cc_from} =")
            st.markdown(f"### {res:,.2f} {cc_to}")
            st.caption(f"Rate: 1 {cc_from} = {rate:,.4f} {cc_to}")
        else:
            st.error("Conversion failed. Try standard pairs.")

# --- Main App Execution ---

if ticker:
    # 1. Fetch Data
    s_date = start_date if interval == "1d" else None
    e_date = end_date if interval == "1d" else None
    
    data = get_stock_data(ticker, interval, period, s_date, e_date)
    info, news = get_stock_info(ticker)

    if data is not None and len(data) > 0:
        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 2. Calculate Indicators
        data = calculate_technicals(data)

        # 3. Key Metrics
        current_price = data['Close'].iloc[-1]
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            delta = current_price - prev_close
            pct = (delta / prev_close) * 100
        else:
            delta, pct = 0, 0
            
        market_cap = info.get('marketCap', 0)
        volume = info.get('volume', 0)
        currency = info.get('currency', 'USD') # Default to USD if missing

        # --- CURRENCY CONVERSION LOGIC ---
        usd_krw_rate = get_exchange_rate("KRW=X")
        price_display = f"{currency} {current_price:,.2f}"
        
        # If asset is USD and we have a rate, show KRW too
        if currency == 'USD' and usd_krw_rate:
            krw_price = current_price * usd_krw_rate
            price_sub_display = f"(₩{krw_price:,.0f})"
        # If asset is KRW, price is already KRW
        elif currency == 'KRW':
            price_sub_display = "(KRW)"
        else:
            price_sub_display = ""

        # 4. Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        
        # Custom Metric for Dual Currency
        with m1:
            st.metric(f"Price ({ticker})", price_display, f"{delta:,.2f} ({pct:+.2f}%)")
            if price_sub_display:
                st.caption(f"≈ {price_sub_display}")
                
        m2.metric("Market Cap", f"${market_cap/1e9:,.1f}B" if market_cap > 0 else "N/A")
        m3.metric("Volume", f"{volume/1e6:,.1f}M" if volume > 0 else "N/A")
        m4.metric("Sector", info.get('sector', 'N/A'))

        # 5. Tabs
        # Conditional Tabs: Show "Fundamentals" only for Stocks
        tabs_list = ["📈 Chart", "🧠 AI Analysis", "📰 News", "🔢 Raw Data"]
        if market_type == "Stocks":
            tabs_list.insert(3, "📋 Fundamentals")
            
        tabs = st.tabs(tabs_list)

        # --- Tab 1: Chart ---
        with tabs[0]:
            fig = go.Figure()
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                name='Price',
                increasing_line_color='#00C853', decreasing_line_color='#FF3D00'
            ))
            
            # Overlays
            if show_sma:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='#FFA000', width=1.5), name=f'SMA {sma_period}'))
            if show_ema:
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], line=dict(color='#2962FF', width=1.5), name=f'EMA {ema_period}'))
            if show_bb:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper BB'))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Lower BB', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))

            fig.update_layout(
                height=600,
                template="plotly_white",
                title_text=f"{ticker} Price Action ({interval})",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                yaxis=dict(title=f'Price ({currency})'),
                xaxis=dict(title='Time')
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 2: AI Analysis (Restored Forecast) ---
        with tabs[1]:
            col_left, col_right = st.columns([1, 1])
            
            # Fear & Greed + News Sentiment
            fg_score, fg_label = get_fear_and_greed_proxy()
            pos, neg, neu, news_label = analyze_news_sentiment(news)
            
            with col_left:
                st.subheader("Market Sentiment")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=fg_score,
                    title={'text': f"Fear & Greed: {fg_label}"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#212529"},
                           'steps': [
                               {'range': [0, 45], 'color': "#FF5252"},
                               {'range': [45, 55], 'color': "#FFD740"},
                               {'range': [55, 100], 'color': "#69F0AE"}]}
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_right:
                st.subheader("News Sentiment")
                df_sent = pd.DataFrame({'Type': ['Pos', 'Neu', 'Neg'], 'Count': [pos, neu, neg]})
                fig_bar = go.Figure(go.Bar(
                    x=df_sent['Type'], y=df_sent['Count'],
                    marker_color=['#69F0AE', '#FFD740', '#FF5252']
                ))
                fig_bar.update_layout(height=250, title="Headline Analysis", margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            # AI Report
            st.markdown("---")
            if len(data) > sma_period:
                report = generate_ai_report(ticker, current_price, data['SMA'].iloc[-1], data['RSI'].iloc[-1], fg_score, fg_label, news_label)
                st.markdown(f"""<div class="ai-analysis-box">{report.replace(chr(10), '<br>')}</div>""", unsafe_allow_html=True)
            
            # --- FORECAST SECTION ---
            st.markdown("### 🔮 Price Forecast (Linear Regression)")
            st.info("Projection based on the trend of the selected historical data.")
            
            if len(data) > 30:
                # Prepare data for ML
                df_ml = data[['Close']].dropna().reset_index()
                df_ml['Ordinal'] = df_ml.index  # Use integer index for regression
                
                X = df_ml[['Ordinal']].values
                y = df_ml['Close'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict next 30 steps
                last_idx = df_ml['Ordinal'].iloc[-1]
                future_X = np.arange(last_idx + 1, last_idx + 31).reshape(-1, 1)
                future_pred = model.predict(future_X)
                
                # Generate future dates (approximate)
                last_date = df_ml.iloc[-1, 0] # timestamp
                if interval == '1m': delta_time = timedelta(minutes=1)
                elif interval == '5m': delta_time = timedelta(minutes=5)
                elif interval == '1h': delta_time = timedelta(hours=1)
                else: delta_time = timedelta(days=1)
                
                future_dates = [last_date + (i * delta_time) for i in range(1, 31)]
                
                # Plot Forecast
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df_ml.iloc[:, 0], y=y, name='History', line=dict(color='#2962FF')))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, name='Forecast', line=dict(color='#FF3D00', dash='dash')))
                
                fig_pred.update_layout(
                    height=400, template="plotly_white", 
                    title="Trend Projection (Next 30 Candles)",
                    hovermode="x unified",
                    yaxis=dict(title=f'Price ({currency})')
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                forecast_change = future_pred[-1] - current_price
                st.caption(f"Projected price in 30 periods: **{currency} {future_pred[-1]:.2f}** ({forecast_change:+.2f})")
            else:
                st.warning("Not enough data points to generate a reliable forecast.")

        # --- Tab 3: News ---
        with tabs[2]:
            st.subheader(f"Recent Headlines for {ticker}")
            if news:
                for item in news[:10]:
                    title = safe_extract_news_title(item)
                    if not title: title = "No Title Available"
                    
                    # Extract Link
                    link = item.get('link') or item.get('url')
                    if not link and 'clickThroughUrl' in item:
                        if isinstance(item['clickThroughUrl'], dict): link = item['clickThroughUrl'].get('url')
                    if not link: link = f"https://finance.yahoo.com/quote/{ticker}/news"
                    
                    # Publisher & Time
                    pub = item.get('publisher', 'Unknown')
                    ts = item.get('providerPublishTime', 0)
                    time_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else "Recent"
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #0d6efd;">
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #0d6efd; font-weight: 600; font-size: 16px;">{title}</a>
                        <div style="font-size: 12px; color: #6c757d; margin-top: 5px;">
                            {pub} • {time_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news found.")

        # --- Tab 4: Fundamentals (Only for Stocks) ---
        if market_type == "Stocks":
            with tabs[3]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Company Profile**")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                with c2:
                    st.markdown("**Financials**")
                    st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "**Dividend Yield:** N/A")
                    st.write(f"**52 Wk High:** ${info.get('fiftyTwoWeekHigh', 0)}")
                    st.write(f"**52 Wk Low:** ${info.get('fiftyTwoWeekLow', 0)}")
                
                st.markdown("---")
                st.markdown("**Business Summary**")
                st.caption(info.get('longBusinessSummary', 'No summary available.'))

        # --- Tab 5: Raw Data ---
        # Adjust index based on whether Fundamentals tab exists
        raw_tab_idx = 4 if market_type == "Stocks" else 3
        with tabs[raw_tab_idx]:
            st.subheader("🔢 High-Precision Data View")
            st.caption("Detailed view of all price points and calculated technical indicators.")
            
            # Reorder columns for readability
            cols_to_show = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA', 'EMA', 'BB_Upper', 'BB_Lower']
            # Filter columns that actually exist in the dataframe
            cols_existing = [c for c in cols_to_show if c in data.columns]
            
            # Display Dataframe
            st.dataframe(
                data[cols_existing].style.format("{:.2f}"), 
                use_container_width=True, 
                height=500
            )
            
            # Download Button
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{ticker}_data.csv",
                mime="text/csv",
            )
            
            st.markdown("### Descriptive Statistics")
            st.dataframe(data.describe(), use_container_width=True)

    else:
        st.error(f"Could not retrieve data for {ticker}. The stock might be delisted or the API is rate-limited.")

else:
    st.info("👈 Please select an asset in the sidebar to begin.")
