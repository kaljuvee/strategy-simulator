#!/usr/bin/env python3
"""
Command-Line Trading Script
Execute trading strategies from the command line for scheduled/automated trading
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import utilities
from utils.alpaca_util import AlpacaAPI
from utils.alpaca_util import is_market_open as alpaca_market_open
from utils.backtester_util import backtest_buy_the_dip, backtest_vix_strategy
from utils.eodhd_util import get_real_time_price, get_historical_data
import time
from datetime import datetime, timedelta


def get_alpaca_client(mode='paper'):
    """Initialize Alpaca client"""
    if mode == 'paper':
        api_key = os.getenv('ALPACA_PAPER_API_KEY')
        secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')
    else:
        api_key = os.getenv('ALPACA_LIVE_API_KEY')
        secret_key = os.getenv('ALPACA_LIVE_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError(f"{mode.upper()} API keys not found in environment")
    
    return AlpacaAPI(api_key=api_key, secret_key=secret_key, paper=(mode == 'paper'))


def execute_buy_the_dip_strategy(client, symbols, capital_per_trade=1000, dip_threshold=5.0, use_intraday=True, skip_if_position=True):
    """
    Execute buy-the-dip strategy
    
    Args:
        client: AlpacaAPI client
        symbols: List of stock symbols
        capital_per_trade: Capital to allocate per trade
        dip_threshold: Percentage dip to trigger buy
    """
    logger.info(f"Executing buy-the-dip strategy for {len(symbols)} symbols")
    
    trades_executed = 0
    
    for symbol in symbols:
        try:
            # Get recent price data (last ~30 days) using EODHD
            from_date = (datetime.utcnow() - timedelta(days=40)).strftime("%Y-%m-%d")
            hist = get_historical_data(symbol, from_date=from_date)
            
            if hist.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            # Calculate dip
            recent_high = float(hist['High'].tail(20).max())
            # Prefer intraday price via EODHD for "current" price if enabled
            current_price = None
            if use_intraday:
                current_price = get_real_time_price(symbol) or None
            if current_price is None:
                current_price = float(hist['Close'].iloc[-1])
            dip_pct = ((recent_high - current_price) / recent_high) * 100
            
            logger.info(f"{symbol}: Recent high ${recent_high:.2f}, Current ${current_price:.2f}, Dip {dip_pct:.2f}%")
            
            # Check if dip threshold met
            if dip_pct >= dip_threshold:
                # Optional: skip if we already have a position in this symbol
                if skip_if_position and client:
                    try:
                        pos = client.get_position(symbol)
                        if isinstance(pos, dict) and 'error' not in pos:
                            logger.info(f"Skipping {symbol}: existing position detected")
                            continue
                    except Exception:
                        pass
                # Calculate quantity
                qty = int(capital_per_trade / current_price)
                
                if qty > 0:
                    logger.info(f"BUY SIGNAL: {symbol} - Dip {dip_pct:.2f}% >= {dip_threshold}%")
                    
                    # Place order
                    result = client.create_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    if 'error' in result:
                        logger.error(f"Order failed for {symbol}: {result['error']}")
                    else:
                        logger.info(f"✅ Order placed: BUY {qty} {symbol} @ market")
                        trades_executed += 1
                else:
                    logger.warning(f"Quantity too small for {symbol} (${current_price:.2f})")
            else:
                logger.debug(f"No signal for {symbol} (dip {dip_pct:.2f}% < {dip_threshold}%)")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    logger.info(f"Strategy execution complete. Trades executed: {trades_executed}")
    return trades_executed


def execute_vix_strategy(client, symbols, capital_per_trade=1000, vix_threshold=20.0):
    """
    Execute VIX-based strategy
    
    Args:
        client: AlpacaAPI client
        symbols: List of stock symbols
        capital_per_trade: Capital to allocate per trade
        vix_threshold: VIX level to trigger trades
    """
    logger.info(f"Executing VIX strategy for {len(symbols)} symbols")
    
    try:
        # Get VIX data
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(period='1d')
        
        if vix_hist.empty:
            logger.error("Could not fetch VIX data")
            return 0
        
        current_vix = vix_hist['Close'].iloc[-1]
        logger.info(f"Current VIX: {current_vix:.2f}")
        
        if current_vix < vix_threshold:
            logger.info(f"VIX {current_vix:.2f} < {vix_threshold}, no trades executed")
            return 0
        
        logger.info(f"VIX {current_vix:.2f} >= {vix_threshold}, executing trades")
        
        trades_executed = 0
        
        for symbol in symbols:
            try:
                # Get current price
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                
                if hist.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                current_price = hist['Close'].iloc[-1]
                qty = int(capital_per_trade / current_price)
                
                if qty > 0:
                    logger.info(f"BUY SIGNAL: {symbol} @ ${current_price:.2f} (VIX={current_vix:.2f})")
                    
                    # Place order
                    result = client.create_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    if 'error' in result:
                        logger.error(f"Order failed for {symbol}: {result['error']}")
                    else:
                        logger.info(f"✅ Order placed: BUY {qty} {symbol} @ market")
                        trades_executed += 1
                else:
                    logger.warning(f"Quantity too small for {symbol} (${current_price:.2f})")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        logger.info(f"Strategy execution complete. Trades executed: {trades_executed}")
        return trades_executed
        
    except Exception as e:
        logger.error(f"Error in VIX strategy: {str(e)}")
        return 0


def close_all_positions(client):
    """Close all open positions"""
    logger.info("Closing all positions")
    
    try:
        result = client.close_all_positions(cancel_orders=True)
        
        if 'error' in result:
            logger.error(f"Error closing positions: {result['error']}")
            return False
        else:
            logger.info("✅ All positions closed")
            return True
            
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        return False


def get_account_status(client):
    """Get and log account status"""
    try:
        account = client.get_account()
        
        if 'error' in account:
            logger.error(f"Error fetching account: {account['error']}")
            return None
        
        logger.info("=== Account Status ===")
        logger.info(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        logger.info(f"Cash: ${float(account.get('cash', 0)):,.2f}")
        logger.info(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        logger.info(f"Equity: ${float(account.get('equity', 0)):,.2f}")
        logger.info("=====================")
        
        return account
        
    except Exception as e:
        logger.error(f"Error getting account status: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Command-line trading script for automated strategy execution')
    
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['buy-the-dip', 'vix', 'close-all', 'status'],
                       help='Strategy to execute')
    
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'live'],
                       help='Trading mode (paper or live)')
    
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA',
                       help='Comma-separated list of stock symbols')
    
    parser.add_argument('--capital', type=float, default=1000.0,
                       help='Capital per trade in dollars')
    
    parser.add_argument('--dip-threshold', type=float, default=5.0,
                       help='Dip threshold percentage for buy-the-dip strategy')
    
    parser.add_argument('--vix-threshold', type=float, default=20.0,
                       help='VIX threshold for VIX strategy')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no actual trades)')
    
    parser.add_argument('--loop', action='store_true',
                       help='Continuously run: check market open and poll at interval')
    
    parser.add_argument('--interval', type=int, default=300,
                       help='Polling interval in seconds for --loop mode (default 300)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info("="*60)
    logger.info(f"CLI Trader Started - {datetime.now()}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("="*60)
    
    # Warning for live trading
    if args.mode == 'live' and not args.dry_run:
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY WILL BE USED!")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        import time
        time.sleep(5)
    
    try:
        # Initialize client
        if not args.dry_run:
            client = get_alpaca_client(args.mode)
            logger.info(f"✅ Connected to Alpaca ({args.mode.upper()} mode)")
        else:
            logger.info("DRY RUN MODE - No actual trades will be executed")
            client = None
        
        # Execute strategy (single-run or loop)
        def run_once():
            if args.strategy == 'status':
                if client:
                    get_account_status(client)
                else:
                    logger.info("Cannot get status in dry-run mode")
                    
            elif args.strategy == 'close-all':
                if client:
                    close_all_positions(client)
                else:
                    logger.info("DRY RUN: Would close all positions")
                    
            elif args.strategy == 'buy-the-dip':
                if client:
                    execute_buy_the_dip_strategy(
                        client,
                        symbols,
                        capital_per_trade=args.capital,
                        dip_threshold=args.dip_threshold,
                        use_intraday=True,
                        skip_if_position=True
                    )
                else:
                    logger.info(f"DRY RUN: Would execute buy-the-dip strategy")
                    logger.info(f"Parameters: capital=${args.capital}, dip_threshold={args.dip_threshold}%")
                    
            elif args.strategy == 'vix':
                if client:
                    execute_vix_strategy(
                        client,
                        symbols,
                        capital_per_trade=args.capital,
                        vix_threshold=args.vix_threshold
                    )
                else:
                    logger.info(f"DRY RUN: Would execute VIX strategy")
                    logger.info(f"Parameters: capital=${args.capital}, vix_threshold={args.vix_threshold}")
        
        if args.loop:
            logger.info("Entering continuous loop mode")
            while True:
                try:
                    if args.dry_run or alpaca_market_open():
                        run_once()
                    else:
                        logger.info("Market is closed; sleeping until next interval")
                except Exception as loop_err:
                    logger.error(f"Error in loop: {loop_err}", exc_info=True)
                time.sleep(max(5, args.interval))
        else:
            run_once()
        
        logger.info("="*60)
        logger.info(f"CLI Trader Completed - {datetime.now()}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Execution cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
