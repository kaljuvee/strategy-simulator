"""
Backtesting Utility for Strategy Simulator
Implements various trading strategies with performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf


def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float, 
                     start_date: datetime, end_date: datetime) -> Dict:
    """Calculate backtest performance metrics"""
    
    if trades_df.empty:
        return {
            'total_return': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    final_capital = trades_df['capital_after'].iloc[-1]
    total_pnl = final_capital - initial_capital
    total_return = (total_pnl / initial_capital) * 100
    
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    total_trades = len(trades_df)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate annualized return
    days = (end_date - start_date).days
    years = days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Calculate max drawdown
    equity_curve = trades_df['capital_after'].values
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    returns = trades_df['pnl_pct'].values
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        'total_return': total_return,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


def get_historical_data(symbols: List[str], start_date: datetime, 
                       end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Fetch historical price data for multiple symbols"""
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return data


def backtest_buy_the_dip(symbols: List[str], start_date: datetime, end_date: datetime,
                        initial_capital: float = 10000, position_size: float = 0.1,
                        dip_threshold: float = 0.02, hold_days: int = 1,
                        take_profit: float = 0.01, stop_loss: float = 0.005) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest buy-the-dip strategy
    
    Strategy: Buy when stock drops by dip_threshold from recent high, 
              hold for hold_days or until take_profit/stop_loss hit
    
    Args:
        symbols: List of stock symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        position_size: Fraction of capital to use per trade
        dip_threshold: Percentage drop to trigger buy (e.g., 0.02 = 2%)
        hold_days: Days to hold position
        take_profit: Take profit percentage (e.g., 0.01 = 1%)
        stop_loss: Stop loss percentage (e.g., 0.005 = 0.5%)
    
    Returns:
        Tuple of (trades_df, metrics_dict)
    """
    
    # Fetch historical data
    price_data = get_historical_data(symbols, start_date - timedelta(days=30), end_date)
    
    if not price_data:
        return None
    
    trades = []
    capital = initial_capital
    
    # Iterate through each trading day
    current_date = start_date
    while current_date <= end_date:
        
        for symbol in symbols:
            if symbol not in price_data:
                continue
            
            df = price_data[symbol]
            
            # Get data up to current date
            historical = df[df.index <= current_date]
            if len(historical) < 20:  # Need at least 20 days of history
                continue
            
            # Calculate 20-day high
            recent_high = float(historical['High'].tail(20).max())
            current_price = float(historical['Close'].iloc[-1])
            
            # Check if we have a dip
            dip_pct = (recent_high - current_price) / recent_high
            
            if dip_pct >= dip_threshold:
                # Enter trade
                entry_price = current_price
                entry_time = current_date
                shares = int((capital * position_size) / entry_price)
                
                if shares == 0:
                    continue
                
                # Calculate target and stop prices
                target_price = entry_price * (1 + take_profit)
                stop_price = entry_price * (1 - stop_loss)
                
                # Simulate holding period
                exit_time = entry_time + timedelta(days=hold_days)
                if exit_time > end_date:
                    exit_time = end_date
                
                # Get exit data
                future_data = df[(df.index > entry_time) & (df.index <= exit_time)]
                
                if future_data.empty:
                    continue
                
                # Check for take profit or stop loss
                hit_target = False
                hit_stop = False
                exit_price = None
                actual_exit_time = None
                
                for idx, row in future_data.iterrows():
                    if float(row['High']) >= target_price:
                        exit_price = target_price
                        actual_exit_time = idx
                        hit_target = True
                        break
                    elif float(row['Low']) <= stop_price:
                        exit_price = stop_price
                        actual_exit_time = idx
                        hit_stop = True
                        break
                
                # If neither hit, exit at end of hold period
                if exit_price is None:
                    exit_price = float(future_data['Close'].iloc[-1])
                    actual_exit_time = future_data.index[-1]
                
                # Calculate P&L
                pnl = (exit_price - entry_price) * shares
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                capital += pnl
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': actual_exit_time,
                    'ticker': symbol,
                    'direction': 'long',
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'target_price': target_price,
                    'stop_price': stop_price,
                    'hit_target': hit_target,
                    'hit_stop': hit_stop,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': capital,
                    'dip_pct': dip_pct * 100
                })
        
        current_date += timedelta(days=1)
    
    if not trades:
        return None
    
    # Create trades dataframe
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    metrics = calculate_metrics(trades_df, initial_capital, start_date, end_date)
    
    return trades_df, metrics


def backtest_vix_strategy(symbols: List[str], start_date: datetime, end_date: datetime,
                         initial_capital: float = 10000, position_size: float = 0.1,
                         vix_threshold: float = 20, hold_overnight: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest VIX fear index strategy
    
    Strategy: Buy when VIX > threshold (high fear), hold overnight or until next day
    
    Args:
        symbols: List of stock symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        position_size: Fraction of capital to use per trade
        vix_threshold: VIX level to trigger buy
        hold_overnight: Whether to hold overnight (True) or sell same day (False)
    
    Returns:
        Tuple of (trades_df, metrics_dict)
    """
    
    # Fetch VIX data
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    
    if vix_data.empty:
        return None
    
    # Fetch stock data
    price_data = get_historical_data(symbols, start_date, end_date)
    
    if not price_data:
        return None
    
    trades = []
    capital = initial_capital
    
    # Iterate through VIX data
    for idx, vix_row in vix_data.iterrows():
        vix_close = float(vix_row['Close'])
        
        # Check if VIX exceeds threshold
        if vix_close > vix_threshold:
            trade_date = idx
            
            for symbol in symbols:
                if symbol not in price_data:
                    continue
                
                df = price_data[symbol]
                
                # Get entry price
                if trade_date not in df.index:
                    continue
                
                entry_price = float(df.loc[trade_date, 'Close'])
                entry_time = trade_date
                shares = int((capital * position_size) / entry_price)
                
                if shares == 0:
                    continue
                
                # Determine exit time
                if hold_overnight:
                    exit_time = entry_time + timedelta(days=1)
                else:
                    exit_time = entry_time
                
                # Get exit price
                future_data = df[df.index > entry_time]
                if future_data.empty:
                    continue
                
                exit_row = future_data.iloc[0]
                exit_price = float(exit_row['Close'])
                actual_exit_time = exit_row.name
                
                # Calculate P&L
                pnl = (exit_price - entry_price) * shares
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                capital += pnl
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': actual_exit_time,
                    'ticker': symbol,
                    'direction': 'long',
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'target_price': 0,
                    'stop_price': 0,
                    'hit_target': False,
                    'hit_stop': False,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': capital,
                    'vix_level': vix_close
                })
    
    if not trades:
        return None
    
    # Create trades dataframe
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    metrics = calculate_metrics(trades_df, initial_capital, start_date, end_date)
    
    return trades_df, metrics


def backtest_momentum_strategy(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000,
    position_size_pct: float = 10.0,
    lookback_period: int = 20,
    momentum_threshold: float = 5.0,
    hold_days: int = 5,
    take_profit_pct: Optional[float] = 10.0,
    stop_loss_pct: Optional[float] = 5.0
) -> Dict:
    """
    Backtest momentum trading strategy
    
    Strategy Logic:
    - Buy when stock shows strong upward momentum (price increase > threshold over lookback period)
    - Hold for specified number of days or until take profit/stop loss hit
    - Exit at take profit or stop loss if set
    
    Args:
        symbols: List of stock symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        position_size_pct: Percentage of capital per trade
        lookback_period: Days to look back for momentum calculation
        momentum_threshold: Minimum momentum percentage to trigger buy
        hold_days: Number of days to hold position
        take_profit_pct: Take profit percentage (optional)
        stop_loss_pct: Stop loss percentage (optional)
        
    Returns:
        Dictionary with backtest results and metrics
    """
    
    trades = []
    capital = initial_capital
    
    for symbol in symbols:
        try:
            # Download historical data
            ticker = yf.Ticker(symbol)
            historical = ticker.history(start=start_date, end=end_date)
            
            if historical.empty:
                continue
            
            # Iterate through dates
            for i in range(lookback_period, len(historical)):
                current_date = historical.index[i]
                
                # Calculate momentum
                lookback_start_price = float(historical['Close'].iloc[i - lookback_period])
                current_price = float(historical['Close'].iloc[i])
                momentum_pct = ((current_price - lookback_start_price) / lookback_start_price) * 100
                
                # Check if momentum threshold is met
                if momentum_pct >= momentum_threshold:
                    # Calculate position size
                    position_value = capital * (position_size_pct / 100)
                    shares = int(position_value / current_price)
                    
                    if shares > 0:
                        entry_price = current_price
                        entry_date = current_date
                        
                        # Calculate exit levels
                        target_price = entry_price * (1 + take_profit_pct / 100) if take_profit_pct else None
                        stop_price = entry_price * (1 - stop_loss_pct / 100) if stop_loss_pct else None
                        
                        # Look for exit
                        exit_date = None
                        exit_price = None
                        exit_reason = 'hold_period'
                        
                        # Check future prices for exit
                        future_data = historical.iloc[i+1:min(i+1+hold_days, len(historical))]
                        
                        for j, row in enumerate(future_data.iterrows()):
                            date, data = row
                            
                            # Check take profit
                            if target_price and float(data['High']) >= target_price:
                                exit_date = date
                                exit_price = target_price
                                exit_reason = 'take_profit'
                                break
                            
                            # Check stop loss
                            if stop_price and float(data['Low']) <= stop_price:
                                exit_date = date
                                exit_price = stop_price
                                exit_reason = 'stop_loss'
                                break
                        
                        # If no exit triggered, exit at end of hold period
                        if exit_date is None and len(future_data) > 0:
                            exit_date = future_data.index[-1]
                            exit_price = float(future_data['Close'].iloc[-1])
                        
                        # Record trade if exit found
                        if exit_date and exit_price:
                            pnl = (exit_price - entry_price) * shares
                            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                            capital += pnl
                            
                            # Align columns to UI expectations (ticker/entry_time/exit_time/pnl_pct etc.)
                            trades.append({
                                'ticker': symbol,
                                'entry_time': entry_date,
                                'exit_time': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'shares': shares,
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'capital_after': capital,
                                # Momentum-specific fields
                                'exit_reason': exit_reason,
                                'momentum_pct': momentum_pct,
                                # Placeholder fields used by UI for other strategies
                                'hit_target': exit_reason == 'take_profit',
                                'hit_stop': exit_reason == 'stop_loss',
                                'dip_pct': np.nan
                            })
        
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    if trades_df.empty:
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(trades_df, initial_capital, start_date, end_date)
    
    # Return in (trades_df, metrics) format to match UI expectations
    return trades_df, metrics
