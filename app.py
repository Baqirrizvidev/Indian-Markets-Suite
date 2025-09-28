import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="India Markets Suite", layout="wide")

# ------------------------- Helpers -------------------------

def ensure_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Map plain ticker to Yahoo Finance ticker for NSE/BSE."""
    s = symbol.strip().upper()
    if not s:
        return s
    if s.startswith("^"):  # indices like ^NSEI
        return s
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + (".BO" if exchange == "BSE" else ".NS")

@st.cache_data(ttl=3600, show_spinner=False)
def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame):
        df = df.dropna()
    return df

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_candles(df: pd.DataFrame, symbol: str, sma1=20, sma2=50, show_rsi=True):
    if df.empty:
        st.warning("No data to plot.")
        return

    df = df.copy()
    df["SMA1"] = df["Close"].rolling(sma1).mean()
    df["SMA2"] = df["Close"].rolling(sma2).mean()
    if show_rsi:
        df["RSI"] = compute_rsi(df["Close"])

    rows = 2 if show_rsi else 1
    specs = [[{"type": "xy"}]] if rows == 1 else [[{"type": "xy"}], [{"type": "xy"}]]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, specs=specs)
    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26A69A", decreasing_line_color="#EF5350"), row=1, col=1)
    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA1"], name=f"SMA {sma1}", line=dict(color="#42A5F5", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA2"], name=f"SMA {sma2}", line=dict(color="#AB47BC", width=1)), row=1, col=1)

    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", line=dict(color="#FFA726", width=1)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#BDBDBD", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#BDBDBD", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20), xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def max_drawdown(equity_curve: pd.Series):
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1
    return float(dd.min()), dd

def format_pct(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "-"

# ------------------------- Backtester -------------------------

def backtest_sma(df: pd.DataFrame, short_win=20, long_win=50, cost_bps=10):
    if df.empty or "Close" not in df:
        return None

    data = df.copy()
    data["ret"] = data["Close"].pct_change().fillna(0.0)
    data["sma_s"] = data["Close"].rolling(short_win).mean()
    data["sma_l"] = data["Close"].rolling(long_win).mean()
    data["pos"] = (data["sma_s"] > data["sma_l"]).astype(int)
    data["pos"] = data["pos"].where(data["sma_s"].notna() & data["sma_l"].notna(), 0)
    trade_flag = data["pos"].diff().fillna(0).abs()  # 1 on entries/exits
    tcost = (cost_bps / 1e4) * trade_flag  # cost applied when switching
    strat_ret = data["pos"].shift(1).fillna(0) * data["ret"] - tcost
    bh_ret = data["ret"]

    # Equity curves
    strat_eq = (1 + strat_ret).cumprod()
    bh_eq = (1 + bh_ret).cumprod()

    n = len(data)
    if n < 5:
        return None
    ann_factor = 252 / n * n  # effectively ~252 trading days adjustment
    # CAGR
    strat_cagr = strat_eq.iloc[-1] ** (252 / max(1, len(strat_ret))) - 1
    bh_cagr = bh_eq.iloc[-1] ** (252 / max(1, len(bh_ret))) - 1
    # Sharpe
    strat_sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)
    bh_sharpe = np.sqrt(252) * bh_ret.mean() / (bh_ret.std() + 1e-12)
    # Max drawdown
    strat_mdd, _ = max_drawdown(strat_eq)
    bh_mdd, _ = max_drawdown(bh_eq)

    # Win rate per trade (approx): measure returns on days after a position change until next change
    trades = []
    pos = data["pos"].values
    rets = strat_ret.values
    i = 1
    while i < len(pos):
        if pos[i-1] == 0 and pos[i] == 1:  # entry
            j = i + 1
            cum = 0.0
            while j < len(pos) and pos[j] == 1:
                cum = (1 + cum) * (1 + rets[j]) - 1
                j += 1
            trades.append(cum)
            i = j
        else:
            i += 1
    if trades:
        win_rate = np.mean([1 if x > 0 else 0 for x in trades])
        avg_trade = np.mean(trades)
    else:
        win_rate = np.nan
        avg_trade = np.nan

    return {
        "data": data,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "strat_eq": strat_eq,
        "bh_eq": bh_eq,
        "metrics": {
            "Strategy CAGR": strat_cagr,
            "Buy&Hold CAGR": bh_cagr,
            "Strategy Sharpe": strat_sharpe,
            "Buy&Hold Sharpe": bh_sharpe,
            "Strategy Max DD": strat_mdd,
            "Buy&Hold Max DD": bh_mdd,
            "Win rate": win_rate,
            "Avg trade return": avg_trade
        }
    }

# ------------------------- Portfolio (FIFO) -------------------------

def analyze_portfolio(tx: pd.DataFrame, default_exchange="NSE"):
    """
    tx columns: date, symbol, qty, price, fees
    qty > 0 buy, qty < 0 sell. Prices in INR.
    """
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx.sort_values("date")
    tx["symbol"] = tx["symbol"].astype(str)

    results = []
    realized_total = 0.0
    lots_by_symbol = {}

    for _, row in tx.iterrows():
        sym_raw = row["symbol"].strip()
        sym = ensure_symbol(sym_raw, default_exchange) if not sym_raw.endswith((".NS", ".BO")) else sym_raw
        qty = float(row["qty"])
        price = float(row["price"])
        fees = float(row.get("fees", 0.0))
        lots = lots_by_symbol.setdefault(sym, [])

        if qty > 0:  # BUY
            eff_price = price + (fees / qty if qty != 0 else 0.0)
            lots.append({"qty": qty, "price": eff_price})
        elif qty < 0:  # SELL
            sell_qty = -qty
            remaining = sell_qty
            realized = 0.0
            while remaining > 0 and lots:
                lot = lots[0]
                take = min(remaining, lot["qty"])
                realized += take * (price - lot["price"])
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= 0:
                    lots.pop(0)
            realized -= fees  # subtract sell fees
            realized_total += realized
        else:
            pass  # qty == 0 ignore

    # Current holdings and valuation
    holdings = []
    for sym, lots in lots_by_symbol.items():
        total_qty = sum(l["qty"] for l in lots)
        if total_qty <= 0:
            continue
        avg_cost = sum(l["qty"] * l["price"] for l in lots) / total_qty
        holdings.append({"symbol": sym, "qty": total_qty, "avg_cost": avg_cost})

    if holdings:
        # Fetch last close for all symbols
        tickers = " ".join([h["symbol"] for h in holdings])
        prices = {}
        try:
            data = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                for sym in [h["symbol"] for h in holdings]:
                    try:
                        close_series = data[sym]["Close"].dropna()
                        if not close_series.empty:
                            prices[sym] = float(close_series.iloc[-1])
                    except Exception:
                        pass
            else:
                close_series = data["Close"].dropna()
                if not close_series.empty:
                    sym_single = holdings[0]["symbol"]
                    prices[sym_single] = float(close_series.iloc[-1])
        except Exception:
            pass

        for h in holdings:
            last_price = prices.get(h["symbol"], np.nan)
            h["last_price"] = last_price
            h["value"] = h["qty"] * last_price if not np.isnan(last_price) else np.nan
            h["unrealized_pnl"] = (last_price - h["avg_cost"]) * h["qty"] if not np.isnan(last_price) else np.nan

    holdings_df = pd.DataFrame(holdings) if holdings else pd.DataFrame(columns=["symbol", "qty", "avg_cost", "last_price", "value", "unrealized_pnl"])
    totals = {
        "realized_pnl": realized_total,
        "market_value": float(holdings_df["value"].sum()) if not holdings_df.empty else 0.0,
        "unrealized_pnl": float(holdings_df["unrealized_pnl"].sum()) if not holdings_df.empty else 0.0
    }
    return holdings_df, totals

# ------------------------- UI: Sidebar -------------------------

st.sidebar.title("India Markets Suite 🇮🇳")
page = st.sidebar.radio("Navigate", ["Dashboard", "Backtester", "Portfolio", "Options (optional)"])
default_exchange = st.sidebar.selectbox("Default Exchange", ["NSE", "BSE"], index=0)

# ------------------------- Dashboard -------------------------

if page == "Dashboard":
    st.title("Market Dashboard")

    st.caption("Tip: NSE tickers end with .NS (e.g., RELIANCE.NS), BSE tickers with .BO (e.g., 500325.BO). We’ll auto-suffix if missing.")

    # Watchlist
    st.subheader("Watchlist")
    default_watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "KOTAKBANK.NS"]
    wl_text = st.text_input("Tickers (comma-separated)", ", ".join(default_watchlist))
    symbols = [s.strip().upper() for s in wl_text.split(",") if s.strip()]
    symbols = [ensure_symbol(s, default_exchange) if not s.endswith((".NS", ".BO", "^")) else s for s in symbols]

    # Build watchlist table
    rows = []
    for sym in symbols[:15]:  # limit to 15 for speed
        try:
            df_1m = get_history(sym, period="1mo", interval="1d")
            if df_1m.empty or len(df_1m) < 2:
                continue
            last = float(df_1m["Close"].iloc[-1])
            prev = float(df_1m["Close"].iloc[-2])
            change = last - prev
            pct = change / prev if prev else np.nan
            df_1y = get_history(sym, period="1y", interval="1d")
            hi52 = float(df_1y["Close"].max()) if not df_1y.empty else np.nan
            lo52 = float(df_1y["Close"].min()) if not df_1y.empty else np.nan
            rows.append({"Symbol": sym, "LTP": last, "Change": change, "Change %": pct, "52w High": hi52, "52w Low": lo52})
        except Exception:
            pass
    if rows:
        wl_df = pd.DataFrame(rows).sort_values("Change %", ascending=False)
        wl_df["Change %"] = wl_df["Change %"].apply(format_pct)
        st.dataframe(wl_df, use_container_width=True)
    else:
        st.info("Enter valid tickers to see the watchlist.")

    # Chart
    st.subheader("Chart")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        sym_in = st.text_input("Symbol to chart", "RELIANCE")
    with col2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    with col3:
        sma1 = st.number_input("SMA Fast", min_value=5, max_value=100, value=20, step=1)
    sma2 = st.slider("SMA Slow", min_value=sma1+5, max_value=200, value=max(50, sma1+30), step=5)

    sym_plot = ensure_symbol(sym_in, default_exchange) if not sym_in.upper().endswith((".NS", ".BO")) else sym_in.upper()
    df_plot = get_history(sym_plot, period=period, interval="1d")
    if df_plot.empty:
        st.error(f"No data for {sym_plot}. Try a different symbol or exchange.")
    else:
        # small header metrics
        last = float(df_plot["Close"].iloc[-1])
        prev = float(df_plot["Close"].iloc[-2]) if len(df_plot) > 1 else last
        st.metric(label=f"{sym_plot} Last Close", value=f"{last:,.2f} INR", delta=f"{last - prev:,.2f} ({(last-prev)/prev*100:.2f}%)")
        plot_candles(df_plot, sym_plot, sma1=sma1, sma2=sma2, show_rsi=True)

# ------------------------- Backtester -------------------------

elif page == "Backtester":
    st.title("Strategy Backtester (SMA Crossover)")
    sym_bt = st.text_input("Symbol", "RELIANCE")
    sym_bt = ensure_symbol(sym_bt, default_exchange) if not sym_bt.upper().endswith((".NS", ".BO")) else sym_bt.upper()
    period = st.selectbox("History", ["2y", "5y", "max"], index=1)
    short_win = st.number_input("Short SMA", min_value=5, max_value=200, value=20, step=1)
    long_win = st.number_input("Long SMA", min_value=10, max_value=400, value=50, step=5)
    if long_win <= short_win:
        st.warning("Long SMA should be greater than Short SMA.")
    cost_bps = st.slider("Transaction cost (bps per entry/exit)", min_value=0, max_value=50, value=10, step=1,
                         help="1 basis point = 0.01%. Applied each time the position changes.")

    if st.button("Run Backtest"):
        df_bt = get_history(sym_bt, period=period, interval="1d")
        if df_bt.empty or len(df_bt) < long_win + 5:
            st.error("Not enough data to backtest. Try a longer period or another symbol.")
        else:
            res = backtest_sma(df_bt, short_win, long_win, cost_bps)
            if res is None:
                st.error("Backtest failed. Try different parameters.")
            else:
                m = res["metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Strategy CAGR", format_pct(m["Strategy CAGR"]))
                c2.metric("Buy&Hold CAGR", format_pct(m["Buy&Hold CAGR"]))
                c3.metric("Strategy Sharpe", f"{m['Strategy Sharpe']:.2f}")
                c4.metric("Max Drawdown (Strat)", format_pct(m["Strategy Max DD"]))

                # Equity curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["strat_eq"].index, y=res["strat_eq"], name="Strategy", line=dict(color="#42A5F5")))
                fig.add_trace(go.Scatter(x=res["bh_eq"].index, y=res["bh_eq"], name="Buy & Hold", line=dict(color="#9CCC65")))
                fig.update_layout(title=f"Equity Curve: {sym_bt}", yaxis_title="Growth of 1", template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Show signal overlay
                st.subheader("Signals on Price")
                data = res["data"].copy()
                fig2 = make_subplots(rows=1, cols=1)
                fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#424242")))
                fig2.add_trace(go.Scatter(x=data.index, y=data["sma_s"], name=f"SMA {short_win}", line=dict(color="#42A5F5", width=1)))
                fig2.add_trace(go.Scatter(x=data.index, y=data["sma_l"], name=f"SMA {long_win}", line=dict(color="#AB47BC", width=1)))
                # Mark entries/exits
                trades = data["pos"].diff().fillna(0)
                entries = data.index[trades == 1]
                exits = data.index[trades == -1]
                fig2.add_trace(go.Scatter(x=entries, y=data.loc[entries, "Close"], mode="markers", name="Entry", marker=dict(color="green", size=8)))
                fig2.add_trace(go.Scatter(x=exits, y=data.loc[exits, "Close"], mode="markers", name="Exit", marker=dict(color="red", size=8)))
                fig2.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig2, use_container_width=True)

# ------------------------- Portfolio -------------------------

elif page == "Portfolio":
    st.title("Portfolio Tracker (FIFO)")

    st.write("Upload transactions CSV with columns: date, symbol, qty, price, fees")
    st.caption("Example: qty > 0 = buy, qty < 0 = sell. Symbols can be plain (e.g., RELIANCE) or suffixed (RELIANCE.NS).")

    sample = """date,symbol,qty,price,fees
2024-04-01,RELIANCE,5,2900,20
2024-06-01,RELIANCE,-2,3000,15
2024-05-10,INFY,10,1450,15
2024-07-20,INFY,-5,1600,10
"""
    st.download_button("Download sample CSV", data=sample, file_name="sample_transactions.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    persist = st.checkbox("Save uploaded file as portfolio.csv (local)", value=False)

    tx_df = None
    if uploaded is not None:
        tx_df = pd.read_csv(uploaded)
        if persist:
            with open("portfolio.csv", "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("Saved to portfolio.csv")
    elif st.button("Load existing portfolio.csv (if present)"):
        try:
            tx_df = pd.read_csv("portfolio.csv")
        except Exception:
            st.error("portfolio.csv not found or invalid.")

    if tx_df is not None:
        required_cols = {"date", "symbol", "qty", "price"}
        if not required_cols.issubset(set(c.lower() for c in tx_df.columns)):
            st.error("CSV must include date, symbol, qty, price (fees optional).")
        else:
            # Normalize column names
            tx_df.columns = [c.lower() for c in tx_df.columns]
            if "fees" not in tx_df.columns:
                tx_df["fees"] = 0.0
            holdings_df, totals = analyze_portfolio(tx_df, default_exchange)
            st.subheader("Holdings")
            if holdings_df.empty:
                st.info("No current holdings.")
            else:
                show_df = holdings_df.copy()
                for col in ["qty", "avg_cost", "last_price", "value", "unrealized_pnl"]:
                    if col in show_df.columns:
                        show_df[col] = show_df[col].round(2)
                st.dataframe(show_df, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Market Value", f"₹{totals['market_value']:,.2f}")
            c2.metric("Unrealized P&L", f"₹{totals['unrealized_pnl']:,.2f}")
            c3.metric("Realized P&L", f"₹{totals['realized_pnl']:,.2f}")

# ------------------------- Options (optional) -------------------------

elif page == "Options (optional)":
    st.title("NSE Options Chain (Optional)")
    st.write("For options, install nsepython (unofficial, may break):")
    st.code("pip install nsepython")
    try:
        from nsepython import nse_optionchain_scrapper  # noqa: F401
        st.success("nsepython detected. You can extend this page to fetch option chains.")
        st.info("Example (to implement): chain = nse_optionchain_scrapper('RELIANCE') → parse CE/PE tables.")
    except Exception:
        st.warning("nsepython not installed or not importable. Skipping live options.")

# ------------------------- Footer -------------------------

st.caption("Data: Yahoo Finance (for NSE/BSE tickers). For learning/research only, not investment advice.")