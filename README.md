# Strategy Simulator

A comprehensive Python Streamlit MVP application for backtesting and paper trading various trading strategies via Alpaca Markets API.

## Features

- **Buy The Dip Strategy**: Backtest buy-the-dip strategy on Mag 7 stocks, S&P 500 members, and sector ETFs
- **VIX Fear Index Strategy**: Trade based on market volatility (VIX index)
- **AI Strategy Assistant**: Powered by XAI Grok for strategy development
- **Alpaca Trader**: Paper and live trading interface with scheduling

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
streamlit run Home.py
```

## Paper Trading - Buy the Dip

Run the buy-the-dip strategy against your Alpaca paper trading account using the CLI trader.

1) Configure environment

```bash
cp .env.example .env
# Edit .env and set at least:
# ALPACA_PAPER_API_KEY=your_paper_key
# ALPACA_PAPER_SECRET_KEY=your_paper_secret
# Optional: for intraday prices (recommended for live/loop mode)
# EODHD_API_KEY=your_eodhd_api_key
```

2) Activate your virtual environment (optional, if you use one)

```bash
source .venv/bin/activate
```

3) Execute the strategy (paper mode, one-off)

```bash
python cli_trader.py --strategy buy-the-dip --mode paper
```

- Continuous mode (polls market when open, uses EODHD intraday prices):

```bash
python cli_trader.py --strategy buy-the-dip --mode paper --loop --interval 300
```

- Optional flags:
  - `--symbols AAPL,MSFT,NVDA` Comma-separated list of tickers (default: AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA)
  - `--capital 1000` Capital per trade in USD (default: 1000)
  - `--dip-threshold 5.0` Percent dip from recent high to trigger buys (default: 5.0)
  - `--dry-run` Simulate without placing orders

Examples:

```bash
# Default basket, paper trading
python cli_trader.py --strategy buy-the-dip --mode paper

# Custom symbols and capital
python cli_trader.py --strategy buy-the-dip --mode paper --symbols AAPL,MSFT,NVDA --capital 2000

# Larger dip threshold and dry-run first
python cli_trader.py --strategy buy-the-dip --mode paper --dip-threshold 7.5 --dry-run
```

Logs are written to `trading.log` and the console.

## Documentation

See full documentation in the application.

