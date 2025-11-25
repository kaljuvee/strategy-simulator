"""
Alpaca Trader Page
Paper and live trading with Alpaca Markets API
"""

import streamlit as st
import pandas as pd
from datetime import datetime, time as dt_time
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.alpaca_util import AlpacaAPI

# Page configuration
st.set_page_config(
    page_title="Alpaca Trader",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Alpaca Trader")
st.markdown("Execute trades and manage positions with Alpaca Markets")

# Initialize session state
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = 'paper'
if 'alpaca_client' not in st.session_state:
    st.session_state.alpaca_client = None

# Sidebar - Trading Mode Selection
st.sidebar.header("⚙️ Trading Configuration")

# Trading mode toggle
trading_mode = st.sidebar.radio(
    "Trading Mode",
    options=['paper', 'live'],
    index=0 if st.session_state.trading_mode == 'paper' else 1,
    help="Switch between paper trading (simulated) and live trading (real money)"
)

# Warning for live trading
if trading_mode == 'live':
    st.sidebar.warning("⚠️ **LIVE TRADING MODE** - Real money will be used!")
    st.sidebar.markdown("Make sure you have:")
    st.sidebar.markdown("- Set ALPACA_LIVE_API_KEY")
    st.sidebar.markdown("- Set ALPACA_LIVE_SECRET_KEY")
    st.sidebar.markdown("- Sufficient funds in your account")
else:
    st.sidebar.success("✅ **PAPER TRADING MODE** - Safe simulation environment")

# Update trading mode if changed
if trading_mode != st.session_state.trading_mode:
    st.session_state.trading_mode = trading_mode
    st.session_state.alpaca_client = None  # Reset client
    st.rerun()

# Initialize Alpaca client
try:
    if st.session_state.alpaca_client is None:
        if trading_mode == 'paper':
            api_key = os.getenv('ALPACA_PAPER_API_KEY')
            secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')
        else:
            api_key = os.getenv('ALPACA_LIVE_API_KEY')
            secret_key = os.getenv('ALPACA_LIVE_SECRET_KEY')
        
        if not api_key or not secret_key:
            st.error(f"❌ {trading_mode.upper()} API keys not found. Please set them in .env file.")
            st.stop()
        
        st.session_state.alpaca_client = AlpacaAPI(
            api_key=api_key,
            secret_key=secret_key,
            paper=(trading_mode == 'paper')
        )
        st.sidebar.success(f"✅ Connected to Alpaca ({trading_mode.upper()} mode)")
        
except Exception as e:
    st.error(f"❌ Failed to initialize Alpaca client: {str(e)}")
    st.stop()

client = st.session_state.alpaca_client

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Account", "💼 Positions", "📝 Orders", "🚀 Place Order", "⏰ Schedule Trades"
])

# Tab 1: Account Information
with tab1:
    st.subheader("Account Information")
    
    if st.button("🔄 Refresh Account Info"):
        st.rerun()
    
    try:
        account = client.get_account()
        
        if 'error' in account:
            st.error(f"Error fetching account: {account['error']}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", f"${float(account.get('portfolio_value', 0)):,.2f}")
                st.metric("Cash", f"${float(account.get('cash', 0)):,.2f}")
            
            with col2:
                st.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
                st.metric("Equity", f"${float(account.get('equity', 0)):,.2f}")
            
            with col3:
                st.metric("Long Market Value", f"${float(account.get('long_market_value', 0)):,.2f}")
                st.metric("Short Market Value", f"${float(account.get('short_market_value', 0)):,.2f}")
            
            with col4:
                day_pl = float(account.get('equity', 0)) - float(account.get('last_equity', 0))
                st.metric("Day P&L", f"${day_pl:,.2f}", delta=f"{(day_pl/float(account.get('last_equity', 1))*100):.2f}%")
                st.metric("Pattern Day Trader", "Yes" if account.get('pattern_day_trader') else "No")
            
            # Account details
            st.markdown("---")
            st.markdown("### Account Details")
            
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write(f"**Account Number:** {account.get('account_number', 'N/A')}")
                st.write(f"**Status:** {account.get('status', 'N/A')}")
                st.write(f"**Trading Blocked:** {account.get('trading_blocked', 'N/A')}")
                st.write(f"**Account Blocked:** {account.get('account_blocked', 'N/A')}")
            
            with details_col2:
                st.write(f"**Daytrade Count:** {account.get('daytrade_count', 0)}")
                st.write(f"**Multiplier:** {account.get('multiplier', 'N/A')}")
                st.write(f"**Currency:** {account.get('currency', 'USD')}")
                st.write(f"**Created At:** {account.get('created_at', 'N/A')}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tab 2: Positions
with tab2:
    st.subheader("Current Positions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Refresh Positions"):
            st.rerun()
    with col2:
        if st.button("🗑️ Close All Positions", type="secondary"):
            if st.session_state.get('confirm_close_all'):
                result = client.close_all_positions(cancel_orders=True)
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("All positions closed!")
                    st.session_state.confirm_close_all = False
                    st.rerun()
            else:
                st.session_state.confirm_close_all = True
                st.warning("Click again to confirm closing all positions")
    
    try:
        positions = client.get_positions()
        
        if isinstance(positions, dict) and 'error' in positions:
            st.error(f"Error fetching positions: {positions['error']}")
        elif not positions:
            st.info("No open positions")
        else:
            # Convert to DataFrame
            positions_data = []
            for pos in positions:
                positions_data.append({
                    'Symbol': pos.get('symbol'),
                    'Qty': float(pos.get('qty', 0)),
                    'Avg Entry': f"${float(pos.get('avg_entry_price', 0)):.2f}",
                    'Current Price': f"${float(pos.get('current_price', 0)):.2f}",
                    'Market Value': f"${float(pos.get('market_value', 0)):,.2f}",
                    'P&L': f"${float(pos.get('unrealized_pl', 0)):,.2f}",
                    'P&L %': f"{float(pos.get('unrealized_plpc', 0))*100:.2f}%",
                    'Side': pos.get('side', 'N/A')
                })
            
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Close individual position
            st.markdown("---")
            st.markdown("### Close Position")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                symbol_to_close = st.selectbox(
                    "Select Symbol",
                    options=[p.get('symbol') for p in positions]
                )
            
            with col2:
                close_qty = st.number_input("Quantity (0 = all)", min_value=0.0, value=0.0, step=1.0)
            
            with col3:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Close Position", type="primary"):
                    qty_to_close = None if close_qty == 0 else close_qty
                    result = client.close_position(symbol_to_close, qty=qty_to_close)
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"Position {symbol_to_close} closed!")
                        st.rerun()
                        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tab 3: Orders
with tab3:
    st.subheader("Order Management")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Refresh Orders"):
            st.rerun()
    with col2:
        if st.button("🗑️ Cancel All Orders", type="secondary"):
            result = client.cancel_all_orders()
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("All orders cancelled!")
                st.rerun()
    
    try:
        orders = client.get_orders()
        
        if isinstance(orders, dict) and 'error' in orders:
            st.error(f"Error fetching orders: {orders['error']}")
        elif not orders:
            st.info("No orders")
        else:
            # Convert to DataFrame
            orders_data = []
            for order in orders:
                orders_data.append({
                    'ID': order.get('id', 'N/A')[:8] + '...',
                    'Symbol': order.get('symbol'),
                    'Side': order.get('side'),
                    'Type': order.get('type'),
                    'Qty': float(order.get('qty', 0)),
                    'Filled': float(order.get('filled_qty', 0)),
                    'Status': order.get('status'),
                    'Limit Price': f"${float(order.get('limit_price', 0)):.2f}" if order.get('limit_price') else 'N/A',
                    'Stop Price': f"${float(order.get('stop_price', 0)):.2f}" if order.get('stop_price') else 'N/A',
                    'Created': order.get('created_at', 'N/A')[:19] if order.get('created_at') else 'N/A'
                })
            
            df = pd.DataFrame(orders_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Cancel individual order
            st.markdown("---")
            st.markdown("### Cancel Order")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                order_to_cancel = st.selectbox(
                    "Select Order",
                    options=[f"{o.get('symbol')} - {o.get('side')} {o.get('qty')} @ {o.get('type')} ({o.get('id')[:8]}...)" for o in orders]
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Cancel Order", type="primary"):
                    # Extract order ID from selection
                    order_id = [o.get('id') for o in orders if f"{o.get('symbol')} - {o.get('side')} {o.get('qty')} @ {o.get('type')} ({o.get('id')[:8]}...)" == order_to_cancel][0]
                    result = client.cancel_order(order_id)
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"Order cancelled!")
                        st.rerun()
                        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tab 4: Place Order
with tab4:
    st.subheader("Place New Order")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", help="Stock ticker symbol")
        side = st.selectbox("Side", options=['buy', 'sell'])
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
    
    with col2:
        order_type = st.selectbox(
            "Order Type",
            options=['market', 'limit', 'stop', 'stop_limit'],
            help="Market: Execute immediately at current price\nLimit: Execute at specified price or better\nStop: Execute when price reaches stop price\nStop Limit: Limit order triggered at stop price"
        )
        time_in_force = st.selectbox(
            "Time in Force",
            options=['day', 'gtc'],
            help="Day: Order valid until end of trading day\nGTC: Good till cancelled"
        )
    
    # Additional fields based on order type
    limit_price = None
    stop_price = None
    
    if order_type in ['limit', 'stop_limit']:
        limit_price = st.number_input("Limit Price", min_value=0.01, value=100.0, step=0.01)
    
    if order_type in ['stop', 'stop_limit']:
        stop_price = st.number_input("Stop Price", min_value=0.01, value=100.0, step=0.01)
    
    # Order preview
    st.markdown("---")
    st.markdown("### Order Preview")
    
    preview_text = f"**{side.upper()}** {qty} shares of **{symbol}** at **{order_type.upper()}**"
    if limit_price:
        preview_text += f" with limit price **${limit_price:.2f}**"
    if stop_price:
        preview_text += f" with stop price **${stop_price:.2f}**"
    preview_text += f" ({time_in_force.upper()})"
    
    st.markdown(preview_text)
    
    # Place order button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Place Order", type="primary", use_container_width=True):
            try:
                kwargs = {}
                if limit_price:
                    kwargs['limit_price'] = limit_price
                if stop_price:
                    kwargs['stop_price'] = stop_price
                
                result = client.create_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    **kwargs
                )
                
                if 'error' in result:
                    st.error(f"❌ Order failed: {result['error']}")
                else:
                    st.success(f"✅ Order placed successfully!")
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Error placing order: {str(e)}")
    
    with col2:
        if st.button("Clear Form", use_container_width=True):
            st.rerun()

# Tab 5: Schedule Trades
with tab5:
    st.subheader("Schedule Trading Times")
    
    st.markdown("""
    Schedule when your trading strategies should run. This allows you to:
    - Set specific times for strategy execution
    - Run backtests and place orders automatically
    - Manage multiple schedules for different strategies
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trading Schedule")
        
        schedule_enabled = st.checkbox("Enable Scheduled Trading", value=False)
        
        if schedule_enabled:
            start_time = st.time_input("Trading Start Time", value=dt_time(9, 30))
            end_time = st.time_input("Trading End Time", value=dt_time(16, 0))
            
            trading_days = st.multiselect(
                "Trading Days",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            )
            
            strategy_to_run = st.selectbox(
                "Strategy to Execute",
                options=['buy-the-dip', 'vix-strategy', 'custom']
            )
            
            if st.button("💾 Save Schedule", type="primary"):
                st.success("Schedule saved! Use the command-line script to run scheduled trades.")
                st.info(f"Schedule: {start_time} - {end_time} on {', '.join(trading_days)}")
    
    with col2:
        st.markdown("### Command Line Execution")
        
        st.markdown("""
        To run trades from the command line or via cron:
        
        ```bash
        # Run a specific strategy
        python cli_trader.py --strategy buy-the-dip --mode paper
        
        # Run with custom parameters
        python cli_trader.py --strategy vix --mode paper --symbols AAPL,MSFT
        
        # Schedule with cron (edit crontab -e)
        # Run every day at 9:30 AM
        30 9 * * 1-5 cd /path/to/strategy-simulator && python cli_trader.py --strategy buy-the-dip
        
        # Run every hour during trading hours
        0 9-16 * * 1-5 cd /path/to/strategy-simulator && python cli_trader.py --strategy vix
        ```
        """)
        
        st.markdown("### Cron Schedule Examples")
        st.code("""
# Every weekday at 9:30 AM
30 9 * * 1-5 python cli_trader.py --strategy buy-the-dip

# Every hour from 9 AM to 4 PM on weekdays
0 9-16 * * 1-5 python cli_trader.py --strategy vix

# Every 15 minutes during trading hours
*/15 9-16 * * 1-5 python cli_trader.py --strategy custom

# Once a day at market close (4 PM)
0 16 * * 1-5 python cli_trader.py --strategy buy-the-dip --mode paper
        """, language="bash")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
    <p>Alpaca Trader | {trading_mode.upper()} Mode | Connected to Alpaca Markets</p>
    </div>
    """,
    unsafe_allow_html=True
)
