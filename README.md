India Markets Suite — Research, Backtesting & Portfolio for NSE/BSE 🇮🇳
A local Streamlit app for Indian markets. Track a watchlist, explore interactive charts, backtest simple strategies, and analyze a FIFO-based portfolio — all using Yahoo Finance data (via yfinance). No API keys needed.

Highlights

Watchlist: Live-ish snapshot with last traded price, daily change, and 52-week high/low
Charts: Candlesticks with configurable moving averages and RSI
Backtester: SMA crossover with CAGR, Sharpe, Max Drawdown, and equity curves
Portfolio: FIFO realized P&L, live valuation, unrealized P&L from your transactions CSV
Optional: Stub to extend with NSE options chain
Screenshots (suggested)

Dashboard: Watchlist + chart
Backtester: Equity curve vs. buy-and-hold
Portfolio: Holdings summary and P&L
What you’ll need

Python 3.9 or newer
VS Code (recommended) with the Python extension
Internet access for market data (yfinance)
Setup in brief

Prepare a clean Python environment (virtual environment recommended).
Install the following Python packages: streamlit, yfinance, pandas, numpy, plotly.
Place the application file (app.py) in your project folder.
Launch Streamlit and open the app in your browser (it typically runs on localhost port 8501).
How to use

Dashboard
Watchlist
Enter NSE/BSE tickers separated by commas.
NSE examples: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, KOTAKBANK.NS
BSE examples: 500325.BO (Reliance), 532540.BO (TCS)
Indices: ^NSEI (Nifty 50), ^NSEBANK (Nifty Bank), ^BSESN (Sensex)
If you type a plain symbol (for example, RELIANCE), the app auto-suffixes using your Default Exchange (set in the sidebar).
Interactive chart
Candlestick view with two SMAs (configurable) and RSI(14)
Select period (6 months to 5 years) and see last close with daily change
Backtester (SMA crossover)
Inputs
Symbol (auto-suffixing supported)
History span (2 years, 5 years, or maximum available)
Short and long SMA windows (for example, 20 and 50)
Transaction cost in basis points (applied on each position flip)
Outputs
Strategy vs. Buy-and-Hold: CAGR, Sharpe ratio, and Max Drawdown
Equity curves and an overlay of entry/exit signals on price
Notes
Uses auto-adjusted prices (splits/dividends accounted for in the data)
Costs modeled as a basis-point deduction when switching between long and flat
Portfolio (FIFO)
Upload a transactions CSV with the following columns:
Required: date, symbol, qty, price
Optional: fees (per trade)
Quantity sign convention: positive = buy, negative = sell
Example structure (sample values)
Row 1: 2024-04-01, RELIANCE, 5, 2900, 20
Row 2: 2024-06-01, RELIANCE, -2, 3000, 15
Row 3: 2024-05-10, INFY, 10, 1450, 15
Row 4: 2024-07-20, INFY, -5, 1600, 10
Results
Realized P&L via FIFO matching
Current holdings: quantity, average cost, last price, market value, unrealized P&L
Summary metrics for market value, unrealized P&L, and realized P&L
Tips
Use plain symbols (for example, RELIANCE) or suffixed symbols (RELIANCE.NS). Plain symbols are auto-suffixed using the selected Default Exchange.
Corporate actions (splits/bonus) aren’t auto-applied to your CSV. Record adjustments explicitly if needed.
Options (optional)
The app includes a stub to integrate an NSE options chain viewer.
You can extend it using community libraries or official APIs as appropriate.
Data and assumptions

Source: Yahoo Finance via yfinance
Prices are auto-adjusted for splits and dividends in historical downloads
Intraday/near-real-time accuracy isn’t guaranteed; data may be delayed
Exchange suffixes: .NS (NSE), .BO (BSE). Indices begin with ^ (for example, ^NSEI)
Metrics explained

CAGR: Annualized growth of the equity curve
Sharpe ratio: Daily return risk-adjusted performance (risk-free rate assumed zero)
Max Drawdown: Peak-to-trough decline in the equity curve
Win rate: Fraction of profitable long-only signal runs (entry to exit)
Transaction costs: Modeled as basis points deducted when the strategy flips position
Project layout

Single-file app (app.py)
Optional persisted file: portfolio.csv (saved from the Portfolio page if you choose to persist)
A virtual environment folder if you use one (not required but recommended)
Configuration and caching

Default Exchange (NSE/BSE) in the sidebar controls auto-suffixing for plain tickers
Historical downloads are cached for faster repeated access
Troubleshooting

Missing packages: Ensure your Python environment is active and install the listed dependencies
No data returned: Confirm the ticker is correct and includes the proper suffix; try a longer period or a different symbol
Backtest errors: Ensure the long SMA is greater than the short SMA and that sufficient history is available
Portfolio mismatches: Validate CSV columns and sign conventions; note that dividends and splits are not auto-applied to your records
Roadmap (ideas to extend)

Additional strategies: MACD, RSI mean reversion, Donchian channel breakouts, momentum rotation
Parameter sweeps and walk-forward analysis
Alerts via email or messaging apps for price or indicator thresholds
Dividends, corporate actions, and tax estimation in the portfolio module
SQLite persistence, multi-portfolio support, and optional authentication
Options: full chain viewer, Greeks, and strategy payoff simulators
Intraday data and alternative data sources
Contributing

Bug fixes, docs improvements, strategy modules, and UI polish are welcome
Keep dependencies minimal and ensure cross-platform compatibility
Note data source terms, licensing, and rate limits when adding integrations
License

MIT License (you may replace with your preferred license if needed)
Disclaimer

For learning and research only; not investment advice
Market data may be delayed or incomplete; use at your own risk
FAQ

Do I need an API key?
No. The app uses yfinance, which does not require an API key.
Can I track BSE tickers?
Yes. Use the .BO suffix or set Default Exchange to BSE so plain symbols are auto-suffixed.
Are dividends handled?
Backtests use adjusted data that reflects dividends. The Portfolio module does not automatically record dividend cash flows.
Does the backtester include slippage and commissions?
It approximates costs using basis points on position flips; adjust the cost setting to reflect your assumptions.
Where is my data stored?
Locally on your machine. Caches are stored temporarily and the portfolio CSV is only saved if you choose to persist it.
If you want, I can also supply a repository-ready version of this README with badges, a changelog, and issue templates—still without any code blocks.




