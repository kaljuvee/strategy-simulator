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

### Configuration

The CLI trader uses `config/parameters.yaml` for all strategy parameters. Edit this file to customize your trading settings:

```yaml
buy_the_dip:
  symbols: "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"  # Stocks to trade
  dip_threshold: 1.0  # Percentage dip to trigger buy (1.0 = 1%)
  take_profit_threshold: 1.0  # Percentage gain to take profit (1.0 = 1%)
  capital_per_trade: 1000.0  # Capital per trade in dollars
  max_position_pct: 5.0  # Max position size as % of buying power

vix:
  symbols: "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA"
  vix_threshold: 20.0
  capital_per_trade: 1000.0

general:
  check_order_status_interval: 60  # Seconds between order status checks
  polling_interval: 300  # Seconds between strategy execution checks
```

**Configuration Parameters:**
- `symbols`: Comma-separated list of stock tickers to trade
- `dip_threshold`: Minimum percentage drop from recent high to trigger a buy order
- `take_profit_threshold`: Target percentage gain to take profit (future feature)
- `capital_per_trade`: Amount of capital to allocate per trade
- `max_position_pct`: Maximum position size as percentage of total buying power
- `check_order_status_interval`: How often to check order status (seconds)
- `polling_interval`: How often to run strategy checks when market is open (seconds)

### Setup

1) Configure environment variables

```bash
cp .env.example .env
# Edit .env and set at least:
# ALPACA_PAPER_API_KEY=your_paper_key
# ALPACA_PAPER_SECRET_KEY=your_paper_secret
# Optional: for intraday prices (recommended for live/loop mode)
# EODHD_API_KEY=your_eodhd_api_key
```

2) Configure trading parameters (optional)

Edit `config/parameters.yaml` to customize:
- Which stocks to trade (`symbols`)
- Dip threshold (`dip_threshold`)
- Take profit threshold (`take_profit_threshold`)
- Capital per trade (`capital_per_trade`)
- Other strategy parameters

3) Activate your virtual environment (optional, if you use one)

```bash
source .venv/bin/activate
```

### Usage

**One-off execution:**

```bash
python tasks/cli_trader.py --strategy buy-the-dip --mode paper
```

**Continuous mode** (polls market when open, uses EODHD intraday prices):

```bash
python tasks/cli_trader.py --strategy buy-the-dip --mode paper --interval 300
```

**Optional CLI flags** (override config file values):
  - `--symbols AAPL,MSFT,NVDA` Override symbols from config
  - `--capital 1000` Override capital per trade
  - `--dip-threshold 1.0` Override dip threshold
  - `--take-profit-threshold 1.0` Override take profit threshold
  - `--dry-run` Simulate without placing orders
  - `--once` Run once and exit (default: run continuously)
  - `--interval 300` Override polling interval (seconds)

**Examples:**

```bash
# Use config file defaults
python tasks/cli_trader.py --strategy buy-the-dip --mode paper

# Override symbols and capital from command line
python tasks/cli_trader.py --strategy buy-the-dip --mode paper --symbols AAPL,MSFT,NVDA --capital 2000

# Test with dry-run first
python tasks/cli_trader.py --strategy buy-the-dip --mode paper --dry-run

# Run once and exit
python tasks/cli_trader.py --strategy buy-the-dip --mode paper --once
```

**Logs:**
- Console output: Real-time trading activity
- `trading.log`: Detailed log file with all operations
- Order status checks: Every 60 seconds (configurable) when orders are pending

## Documentation

See full documentation in the application.

