import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockOracle · Indian Market Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Build & cache model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Initialising model…")
def load_model():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor

    np.random.seed(42)
    n = 8000
    close        = np.random.uniform(50, 8000, n)
    open_        = close * np.random.uniform(0.97, 1.03, n)
    high         = np.maximum(close, open_) * np.random.uniform(1.00, 1.04, n)
    low          = np.minimum(close, open_) * np.random.uniform(0.96, 1.00, n)
    volume       = np.random.randint(50_000, 80_000_000, n).astype(float)
    dow          = np.random.randint(0, 5, n).astype(float)
    month        = np.random.randint(1, 13, n).astype(float)
    year         = np.random.uniform(2015, 2025, n)
    exchanges    = np.random.choice(["NSE", "BSE"], n)
    daily_return = ((close - open_) / np.where(open_ != 0, open_, 1)) * 100
    price_range  = high - low
    avg_price    = (open_ + high + low + close) / 4
    sma7         = avg_price * np.random.uniform(0.98, 1.02, n)
    sma30        = avg_price * np.random.uniform(0.96, 1.04, n)
    future_close = (
        close
        + (close * np.random.normal(0.0003, 0.015, n))
        + np.random.normal(0, close * 0.005, n)
    )

    X = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "day_of_week": dow, "month": month, "year": year,
        "daily_return": daily_return, "price_range": price_range,
        "avg_price": avg_price, "SMA_7": sma7, "SMA_30": sma30,
        "exchange": exchanges,
    })
    y = future_close

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["exchange"])],
        remainder="passthrough",
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            random_state=42, max_iter=300,
            learning_rate=0.05, max_leaf_nodes=31,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline

model = load_model()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --bg:        #080c10;
  --surface:   #0d1117;
  --panel:     #111820;
  --border:    #1e2d3d;
  --accent:    #00d4ff;
  --green:     #00ff9d;
  --red:       #ff4c6a;
  --muted:     #4a6070;
  --text:      #c9d8e8;
  --dim:       #607080;
  --font-ui:   'Syne', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-ui) !important;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridPulse 8s ease-in-out infinite;
}
@keyframes gridPulse { 0%,100%{opacity:.5} 50%{opacity:1} }

[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,.07) 2px, rgba(0,0,0,.07) 4px
  );
}

[data-testid="stMainBlockContainer"],
.main .block-container {
  position: relative; z-index: 1;
  max-width: 900px !important;
  padding: 2rem 1.5rem 4rem !important;
}

p, span, div, label, h1, h2, h3 {
  font-family: var(--font-ui) !important;
  color: var(--text) !important;
}

input[type="number"], input[type="text"],
[data-baseweb="input"] input,
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
}

[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
}
[data-baseweb="popover"] { background: #0d1117 !important; }
li[role="option"] { background: var(--surface) !important; color: var(--text) !important; }
li[role="option"]:hover { background: var(--panel) !important; }

[data-testid="stWidgetLabel"] p, label {
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  font-family: var(--font-ui) !important;
}

[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #00d4ff 0%, #0095cc 100%) !important;
  color: #000 !important;
  font-family: var(--font-ui) !important;
  font-weight: 800 !important;
  font-size: 14px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.75rem 2.5rem !important;
  box-shadow: 0 4px 24px rgba(0,212,255,.35) !important;
  transition: all .25s ease !important;
  width: 100% !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  box-shadow: 0 6px 36px rgba(0,212,255,.55) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--dim) !important;
  font-family: var(--font-ui) !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  transition: all .2s ease !important;
}

[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 9px !important;
  letter-spacing: 2.5px !important;
  color: var(--dim) !important;
  font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  font-size: 20px !important;
  font-weight: 400 !important;
  color: var(--text) !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

@keyframes shimmer  { 0%{background-position:0%} 100%{background-position:200%} }
@keyframes ticker   { from{transform:translateX(0)} to{transform:translateX(-50%)} }
@keyframes blink    { 0%,100%{opacity:1} 50%{opacity:.2} }
@keyframes slideUp  { from{opacity:0;transform:translateY(20px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 2.5rem;">
      <div style="display:inline-flex;align-items:center;gap:8px;font-size:11px;
                  font-weight:700;letter-spacing:4px;text-transform:uppercase;
                  color:#00d4ff;font-family:'Syne',sans-serif;margin-bottom:14px;">
        <span style="width:6px;height:6px;border-radius:50%;background:#00d4ff;
                     box-shadow:0 0 8px #00d4ff;display:inline-block;
                     animation:blink 2s ease-in-out infinite;"></span>
        StockOracle &nbsp;·&nbsp; NSE / BSE
      </div>
      <h1 style="font-size:clamp(2.2rem,5vw,3.8rem);font-weight:800;line-height:1.05;
                 letter-spacing:-2px;color:#fff;margin:0 0 10px;font-family:'Syne',sans-serif;
                 text-shadow:0 0 40px rgba(0,212,255,.2);">
        Next-Day Close<br>
        <span style="background:linear-gradient(90deg,#00d4ff 0%,#7b5fff 50%,#ff6b35 100%);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-size:200%;animation:shimmer 4s linear infinite;">
          Price Predictor
        </span>
      </h1>
      <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#607080;letter-spacing:1px;margin:0;">
        // HistGradientBoosting &nbsp;·&nbsp; scikit-learn Pipeline &nbsp;·&nbsp; Indian Equity Markets
      </p>
    </div>
    <div style="overflow:hidden;white-space:nowrap;border-top:1px solid #1e2d3d;
                border-bottom:1px solid #1e2d3d;padding:7px 0;margin-bottom:2rem;
                background:#0d1117;font-family:'JetBrains Mono',monospace;
                font-size:11px;color:#4a6070;letter-spacing:1px;">
      <span style="display:inline-block;animation:ticker 22s linear infinite;">
        RELIANCE &nbsp;·&nbsp; TCS &nbsp;·&nbsp; HDFCBANK &nbsp;·&nbsp; INFY &nbsp;·&nbsp;
        ICICIBANK &nbsp;·&nbsp; WIPRO &nbsp;·&nbsp; SBIN &nbsp;·&nbsp; HINDUNILVR &nbsp;·&nbsp;
        BAJFINANCE &nbsp;·&nbsp; KOTAKBANK &nbsp;·&nbsp; LT &nbsp;·&nbsp; AXISBANK &nbsp;·&nbsp;
        ASIANPAINT &nbsp;·&nbsp; MARUTI &nbsp;·&nbsp; SUNPHARMA &nbsp;·&nbsp; TATAMOTORS &nbsp;·&nbsp;
        HCLTECH &nbsp;·&nbsp; RELIANCE &nbsp;·&nbsp; TCS &nbsp;·&nbsp; HDFCBANK &nbsp;·&nbsp;
        INFY &nbsp;·&nbsp; ICICIBANK &nbsp;·&nbsp; WIPRO &nbsp;·&nbsp; SBIN &nbsp;·&nbsp;
      </span>
    </div>
    """, unsafe_allow_html=True)


def section_label(number: str, title: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:1.8rem 0 1rem;">
      <span style="font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
                   color:#00d4ff;font-family:'Syne',sans-serif;white-space:nowrap;">
        {number} · {title}
      </span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e2d3d,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


def render_result(pred: float, current: float):
    change     = pred - current
    change_pct = (change / current) * 100
    is_up      = change >= 0
    color      = "#00ff9d" if is_up else "#ff4c6a"
    bg_color   = "rgba(0,255,157,.06)" if is_up else "rgba(255,76,106,.06)"
    border_col = "rgba(0,255,157,.3)"  if is_up else "rgba(255,76,106,.3)"
    arrow      = "↑" if is_up else "↓"
    label      = "BULLISH" if is_up else "BEARISH"
    abs_pct    = abs(change_pct)
    if abs_pct > 3:   signal = "STRONG BUY"  if is_up else "STRONG SELL"
    elif abs_pct > 1: signal = "BUY"          if is_up else "SELL"
    else:             signal = "NEUTRAL"
    sign = "+" if is_up else ""

    st.markdown(f"""
    <div style="background:{bg_color};border:1px solid {border_col};border-radius:14px;
                padding:28px 30px;position:relative;overflow:hidden;
                animation:slideUp .5s cubic-bezier(.34,1.56,.64,1) both;margin-top:1.5rem;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{color},transparent);"></div>
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;margin-bottom:8px;">
        <div>
          <p style="font-size:9px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
                    color:#607080;margin:0 0 8px;font-family:'Syne',sans-serif;">
            Predicted Next-Day Close
          </p>
          <p style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:300;
                    letter-spacing:-2px;line-height:1;margin:0;color:#fff;">
            <span style="font-size:.4em;vertical-align:super;color:#607080;">&#8377;</span>
            {pred:,.2f}
          </p>
        </div>
        <div style="display:inline-flex;align-items:center;gap:6px;padding:8px 18px;
                    border-radius:100px;background:{bg_color};color:{color};
                    border:1px solid {border_col};font-size:14px;font-weight:700;
                    font-family:'JetBrains Mono',monospace;align-self:center;">
          <span style="font-size:20px;">{arrow}</span>{label}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current Close", f"₹{current:,.2f}")
    with m2: st.metric("Change ₹",      f"{sign}₹{change:.2f}")
    with m3: st.metric("Change %",      f"{sign}{change_pct:.4f}%")
    with m4: st.metric("Signal",        signal)

    st.markdown("""
    <p style="margin:14px 0 0;font-family:'JetBrains Mono',monospace;
              font-size:10px;color:#4a6070;letter-spacing:.5px;">
      &#9888; &nbsp;Model estimate only — not financial advice.
    </p>
    """, unsafe_allow_html=True)


def build_features(vals: dict) -> pd.DataFrame:
    o, h, l, c = vals["open"], vals["high"], vals["low"], vals["close"]
    return pd.DataFrame([{
        "open": o, "high": h, "low": l, "close": c,
        "volume":       vals["volume"],
        "day_of_week":  vals["day_of_week"],
        "month":        vals["month"],
        "year":         vals["year"],
        "daily_return": ((c - o) / o) * 100 if o != 0 else 0.0,
        "price_range":  h - l,
        "avg_price":    (o + h + l + c) / 4,
        "SMA_7":        (o + h + l + c) / 4,
        "SMA_30":       (o + h + l + c) / 4,
        "exchange":     vals["exchange"],
    }])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    if "loaded_sample" not in st.session_state:
        st.session_state.loaded_sample = False

    SAMPLE = dict(
        open=2481.50, high=2515.00, low=2462.30, close=2498.75,
        volume=8_200_000, exchange="NSE", day_of_week=2, month=5, year=2025,
    )

    _, col_b, col_c = st.columns([2, 1, 1])
    with col_b:
        if st.button("⚡  Load Sample", key="sample", use_container_width=True):
            st.session_state.loaded_sample = True
            st.rerun()
    with col_c:
        if st.button("✕  Clear", key="clear", use_container_width=True):
            st.session_state.loaded_sample = False
            st.rerun()

    def d(key):        return SAMPLE[key] if st.session_state.loaded_sample else None
    def df_num(key, fallback=0.0): return float(d(key)) if d(key) is not None else fallback

    # ── Section 01 ────────────────────────────────────────────────────────────
    section_label("01", "Market Data")
    c1, c2, c3 = st.columns(3)
    with c1: open_  = st.number_input("Open ₹",  min_value=0.0, value=df_num("open"),  step=0.05, format="%.2f")
    with c2: high   = st.number_input("High ₹",  min_value=0.0, value=df_num("high"),  step=0.05, format="%.2f")
    with c3: low    = st.number_input("Low ₹",   min_value=0.0, value=df_num("low"),   step=0.05, format="%.2f")
    c4, c5, c6 = st.columns(3)
    with c4: close  = st.number_input("Close ₹", min_value=0.0, value=df_num("close"), step=0.05, format="%.2f")
    with c5: volume = st.number_input("Volume",  min_value=0,   value=int(df_num("volume", 0)), step=1000, format="%d")
    with c6:
        exchange_idx = 0 if (d("exchange") or "NSE") == "NSE" else 1
        exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=exchange_idx)

    # ── Section 02 ────────────────────────────────────────────────────────────
    section_label("02", "Date Features")
    now = datetime.now()
    c7, c8, c9 = st.columns(3)

    DAY_OPTIONS = {"0 · Monday":0,"1 · Tuesday":1,"2 · Wednesday":2,"3 · Thursday":3,"4 · Friday":4}
    with c7:
        raw_dow   = min(int(df_num("day_of_week", now.weekday())), 4)
        dow_key   = [k for k,v in DAY_OPTIONS.items() if v == raw_dow][0]
        dow_label = st.selectbox("Day of Week", list(DAY_OPTIONS.keys()), index=list(DAY_OPTIONS.keys()).index(dow_key))
        day_of_week = DAY_OPTIONS[dow_label]

    MONTH_OPTIONS = {"01 · January":1,"02 · February":2,"03 · March":3,"04 · April":4,
                     "05 · May":5,"06 · June":6,"07 · July":7,"08 · August":8,
                     "09 · September":9,"10 · October":10,"11 · November":11,"12 · December":12}
    with c8:
        m_val   = int(df_num("month", now.month))
        m_key   = [k for k,v in MONTH_OPTIONS.items() if v == m_val][0]
        m_label = st.selectbox("Month", list(MONTH_OPTIONS.keys()), index=list(MONTH_OPTIONS.keys()).index(m_key))
        month   = MONTH_OPTIONS[m_label]

    with c9:
        year = st.number_input("Year", min_value=2000, max_value=2099,
                                value=int(df_num("year", now.year)), step=1, format="%d")

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
      <span style="width:6px;height:6px;border-radius:50%;background:#00ff9d;
                   box-shadow:0 0 6px #00ff9d;display:inline-block;animation:blink 2s ease-in-out infinite;"></span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a6070;letter-spacing:1px;">
        Model &nbsp;·&nbsp; HistGradientBoosting &nbsp;·&nbsp; scikit-learn Pipeline &nbsp;·&nbsp; NSE &amp; BSE
      </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Predict Next-Day Close Price", type="primary", use_container_width=True, key="predict"):
        errors = []
        if open_  <= 0: errors.append("Open must be > 0")
        if high   <= 0: errors.append("High must be > 0")
        if low    <= 0: errors.append("Low must be > 0")
        if close  <= 0: errors.append("Close must be > 0")
        if volume <= 0: errors.append("Volume must be > 0")
        if high   < low: errors.append("High must be >= Low")
        if errors:
            for e in errors: st.error(f"⚠  {e}")
        else:
            with st.spinner("Running inference…"):
                df = build_features(dict(open=open_, high=high, low=low, close=close,
                    volume=volume, exchange=exchange,
                    day_of_week=day_of_week, month=month, year=year))
                prediction = float(model.predict(df)[0])
            render_result(prediction, close)

    st.markdown("""
    <div style="text-align:center;margin-top:4rem;font-family:'JetBrains Mono',monospace;
                font-size:11px;color:#4a6070;letter-spacing:.8px;line-height:2;">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                   background:#00ff9d;box-shadow:0 0 6px #00ff9d;
                   vertical-align:middle;margin-right:6px;"></span>
      StockOracle &nbsp;/&nbsp; scikit-learn HistGBM &nbsp;/&nbsp; NSE &middot; BSE
      <br/>&#9888; &nbsp;Not financial advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
