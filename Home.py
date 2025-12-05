"""
Strategy Simulator - Home Page
Main backtesting interface for Buy-The-Dip strategy
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.backtester_util import backtest_buy_the_dip, backtest_momentum_strategy, calculate_buy_and_hold, calculate_single_buy_and_hold
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Strategy Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Strategy Simulator")
st.markdown("Backtest and analyze trading strategies with real market data")

# Sidebar configuration
st.sidebar.header("Strategy Configuration")

# Strategy selection
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["buy-the-dip", "momentum"],
    help="Select the trading strategy to backtest"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Stock Selection")

# Magnificent 7 stocks
MAG7_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Meta": "META"
}

# Additional popular stocks and ETFs
ADDITIONAL_STOCKS = {
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "Salesforce": "CRM",
    "Intel": "INTC",
    "AMD": "AMD",
    "Cisco": "CSCO",
    "Oracle": "ORCL",
    "IBM": "IBM",
    "PayPal": "PYPL",
    "Uber": "UBER",
    "Airbnb": "ABNB",
    "Coinbase": "COIN",
    "Block": "SQ",
    "Shopify": "SHOP",
    "Zoom": "ZM",
    "Palantir": "PLTR",
    "Snowflake": "SNOW",
    "Datadog": "DDOG",
    "CrowdStrike": "CRWD",
    "MongoDB": "MDB"
}

SECTOR_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Consumer Staples": "XLP",
    "Communication Services": "XLC"
}

# Combine all options
all_options = {**MAG7_STOCKS, **ADDITIONAL_STOCKS, **SECTOR_ETFS}

# Default to Mag 7
default_selections = list(MAG7_STOCKS.keys())

selected_names = st.sidebar.multiselect(
    "Select Holdings",
    options=list(all_options.keys()),
    default=default_selections,
    help="Select stocks or ETFs to include in the backtest"
)

# Convert selected names to symbols
selected_symbols = [all_options[name] for name in selected_names]

st.sidebar.markdown("---")
st.sidebar.subheader("Data Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["yfinance"],
    help="Data source for historical prices"
)

# Frequency/interval selection
interval = st.sidebar.selectbox(
    "Data Frequency",
    ["1d", "60m", "30m", "15m", "5m"],
    index=0,
    help="Time interval for price data (intraday requires yfinance)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Parameters")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Capital and position sizing
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Starting capital for the backtest"
)

position_size = st.sidebar.slider(
    "Position Size (%)",
    min_value=1,
    max_value=50,
    value=10,
    help="Percentage of capital to allocate per trade"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

if strategy_type == "buy-the-dip":
    # Buy-the-dip specific parameters
    dip_threshold = st.sidebar.slider(
        "Dip Threshold (%)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Percentage drop from recent high to trigger buy"
    )
    
    hold_days = st.sidebar.number_input(
        "Hold Days",
        min_value=1,
        max_value=30,
        value=1,
        help="Number of days to hold position"
    )
    
    take_profit = st.sidebar.slider(
        "Take Profit (%)",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Percentage gain to take profit"
    )
    
    stop_loss = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Percentage loss to stop out"
    )

elif strategy_type == "momentum":
    # Momentum strategy specific parameters
    lookback_period = st.sidebar.number_input(
        "Lookback Period (days)",
        min_value=5,
        max_value=60,
        value=20,
        help="Number of days to look back for momentum calculation"
    )
    
    momentum_threshold = st.sidebar.slider(
        "Momentum Threshold (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Minimum momentum percentage to trigger buy"
    )
    
    hold_days = st.sidebar.number_input(
        "Hold Days",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of days to hold position"
    )
    
    take_profit = st.sidebar.slider(
        "Take Profit (%)",
        min_value=1.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        help="Percentage gain to take profit"
    )
    
    stop_loss = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Percentage loss to stop out"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Fee Configuration")

include_taf_fees = st.sidebar.checkbox(
    "Include FINRA TAF Fees",
    value=True,
    help="FINRA Trading Activity Fee (TAF): $0.000166 per share (sells only), rounded up to nearest penny, capped at $8.30 per trade"
)

include_cat_fees = st.sidebar.checkbox(
    "Include CAT Fees",
    value=True,
    help="Consolidated Audit Trail (CAT) Fee: $0.0000265 per share (applies to both buys and sells)"
)

st.sidebar.markdown(
    "[Alpaca Fees Documentation](https://alpaca.markets/support/regulatory-fees)",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

with st.sidebar.expander("📋 Fee Details", expanded=False):
    st.markdown("""
    **Regulatory Fees Overview**
    
    Regulatory fees are charged by the US Securities & Exchange Commission (SEC) and Financial Industry Regulatory Authority (FINRA) on all sell orders. Alpaca Securities LLC does not benefit financially from these charges, and they are passed on to the relevant regulatory agencies in full.
    
    ---
    
    **SEC Fee**
    - **Rate:** $0.00 per $1,000,000 of principal
    - **Applies to:** Sells only
    - **Rounding:** Rounded up to the nearest penny
    
    ---
    
    **FINRA Trading Activity Fee (TAF)**
    - **Rate:** $0.000166 per share
    - **Applies to:** Sells only
    - **Application:** Per-trade basis
    - **Rounding:** Rounded up to the nearest penny
    - **Cap:** Maximum $8.30 per trade
    
    ---
    
    **Consolidated Audit Trail (CAT) Fee**
    
    The Consolidated Audit Trail (CAT) is introducing a new fee structure, which Alpaca will be billed for. This fee will be charged per trade and passed on to users and clients, in addition to the existing REG/TAF fee.
    
    The CAT Fee Rate is calculated based on transaction volume and applies to both equity and options trading activities. For options, the fee is typically assessed per executed equivalent share. Since one listed option contract generally represents 100 shares, the CAT fee for options is calculated accordingly.
    
    **Equities:**
    
    Formula: **Fee × Executed Equivalent Shares = CAT Fee**
    
    - **NMS Equities:** $0.0000265 per share (1:1 ratio)
    - **OTC Equities:** $0.0000265 per share (1:0.01 ratio)
    
    **CAT Fee Breakdown:**
    - CAT Fee 2025-2: $0.000009
    - Historical CAT Fee: $0.000013
    - Prospective CAT Cost Recovery Fee 2025-2: $0.0000045
    
    **Total CAT Fee:** $0.0000265 per share
    """)

# Main content area
if len(selected_symbols) == 0:
    st.warning("⚠️ Please select at least one stock or ETF to backtest")
    st.stop()

# Display selected holdings
st.subheader("Selected Holdings")
holdings_df = pd.DataFrame({
    'Name': selected_names,
    'Symbol': selected_symbols
})
st.dataframe(holdings_df, hide_index=True, use_container_width=True)

# Run backtest button
if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        st.stop()
    
    with st.spinner("Running backtest... This may take a moment."):
        try:
            # Run backtest based on strategy type
            if strategy_type == "buy-the-dip":
                results = backtest_buy_the_dip(
                    symbols=selected_symbols,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    initial_capital=initial_capital,
                    position_size=position_size / 100,
                    dip_threshold=dip_threshold / 100,
                    hold_days=hold_days,
                    take_profit=take_profit / 100,
                    stop_loss=stop_loss / 100,
                    interval=interval,
                    data_source=data_source,
                    include_taf_fees=include_taf_fees,
                    include_cat_fees=include_cat_fees
                )
            elif strategy_type == "momentum":
                results = backtest_momentum_strategy(
                    symbols=selected_symbols,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    initial_capital=initial_capital,
                    position_size_pct=position_size,
                    lookback_period=lookback_period,
                    momentum_threshold=momentum_threshold,
                    hold_days=hold_days,
                    take_profit_pct=take_profit,
                    stop_loss_pct=stop_loss,
                    interval=interval,
                    data_source=data_source,
                    include_taf_fees=include_taf_fees,
                    include_cat_fees=include_cat_fees
                )
            
            if results is None:
                st.error("❌ No trades were generated during the backtest period. Try adjusting parameters or date range.")
                st.stop()
            
            trades_df, metrics = results
            
            # Display results
            st.success("✅ Backtest completed successfully!")
            
            st.markdown("---")
            st.header("📊 Performance Metrics")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{metrics['total_return']:.2f}%",
                    delta=f"${metrics['total_pnl']:,.2f}"
                )
                st.metric("Total Trades", metrics['total_trades'])
            
            with col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                st.metric("Winning Trades", metrics['winning_trades'])
            
            with col3:
                st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                st.metric("Losing Trades", metrics['losing_trades'])
            
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            # Calculate buy-and-hold comparison metrics
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            
            spy_dates, spy_values = calculate_single_buy_and_hold('SPY', start_dt, end_dt, initial_capital)
            portfolio_dates, portfolio_values = calculate_buy_and_hold(selected_symbols, start_dt, end_dt, initial_capital)
            
            # Calculate buy-and-hold returns
            spy_return = 0.0
            portfolio_return = 0.0
            
            if not spy_values.empty and len(spy_values) > 0:
                spy_final = spy_values.iloc[-1]
                spy_return = ((spy_final - initial_capital) / initial_capital) * 100
            
            if not portfolio_values.empty and len(portfolio_values) > 0:
                portfolio_final = portfolio_values.iloc[-1]
                portfolio_return = ((portfolio_final - initial_capital) / initial_capital) * 100
            
            # Display comparison metrics
            st.markdown("---")
            st.header("📊 Buy & Hold Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric(
                    "Strategy Return",
                    f"{metrics['total_return']:.2f}%",
                    delta=f"${metrics['total_pnl']:,.2f}"
                )
            
            with comp_col2:
                if portfolio_return != 0.0:
                    delta_portfolio = metrics['total_return'] - portfolio_return
                    st.metric(
                        f"Buy & Hold ({', '.join(selected_symbols[:2])}{'...' if len(selected_symbols) > 2 else ''})",
                        f"{portfolio_return:.2f}%",
                        delta=f"{delta_portfolio:+.2f}% vs Strategy"
                    )
            
            with comp_col3:
                if spy_return != 0.0:
                    delta_spy = metrics['total_return'] - spy_return
                    st.metric(
                        "Buy & Hold (SPY)",
                        f"{spy_return:.2f}%",
                        delta=f"{delta_spy:+.2f}% vs Strategy"
                    )
            
            st.markdown("---")
            st.header("📈 Equity Curve")
            
            # Create equity curve chart
            fig = go.Figure()
            
            # Strategy performance
            fig.add_trace(go.Scatter(
                x=trades_df['entry_time'],
                y=trades_df['capital_after'],
                mode='lines+markers',
                name='Strategy Performance',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Portfolio buy-and-hold
            if not portfolio_values.empty:
                fig.add_trace(go.Scatter(
                    x=portfolio_dates,
                    y=portfolio_values,
                    mode='lines',
                    name=f'Buy & Hold ({", ".join(selected_symbols[:3])}{"..." if len(selected_symbols) > 3 else ""})',
                    line=dict(color='#2ca02c', width=2, dash='dot')
                ))
            
            # SPY buy-and-hold
            if not spy_values.empty:
                fig.add_trace(go.Scatter(
                    x=spy_dates,
                    y=spy_values,
                    mode='lines',
                    name='Buy & Hold (SPY)',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
            
            # Add initial capital line
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
                annotation_position="right"
            )
            
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add fee note if fees are included
            if include_taf_fees or include_cat_fees:
                fee_note = "**Note:** "
                fees_list = []
                if include_taf_fees:
                    fees_list.append("FINRA Trading Activity Fee (TAF)")
                if include_cat_fees:
                    fees_list.append("Consolidated Audit Trail (CAT) fee")
                fee_note += " and ".join(fees_list) + " included in backtest results."
                st.markdown(fee_note)
            
            st.markdown("---")
            st.header("📋 Trade History")
            
            # Format trades dataframe for display
            display_df = trades_df.copy()
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
            display_df['capital_after'] = display_df['capital_after'].apply(lambda x: f"${x:,.2f}")
            if 'dip_pct' in display_df.columns:
                display_df['dip_pct'] = display_df['dip_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            
            # Format fees if included
            if include_taf_fees and 'taf_fee' in display_df.columns:
                display_df['taf_fee'] = display_df['taf_fee'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            
            if include_cat_fees and 'cat_fee' in display_df.columns:
                display_df['cat_fee'] = display_df['cat_fee'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            
            if (include_taf_fees or include_cat_fees) and 'total_fees' in display_df.columns:
                display_df['total_fees'] = display_df['total_fees'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            
            # Select columns to display
            display_columns = [
                'entry_time', 'exit_time', 'ticker', 'shares', 
                'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                'hit_target', 'hit_stop', 'capital_after'
            ]
            
            # Add fee columns if fees were included
            if include_taf_fees and 'taf_fee' in display_df.columns:
                display_columns.append('taf_fee')
            
            if include_cat_fees and 'cat_fee' in display_df.columns:
                display_columns.append('cat_fee')
            
            if (include_taf_fees or include_cat_fees) and 'total_fees' in display_df.columns:
                display_columns.append('total_fees')
            
            # Add dip_pct if it exists (for buy-the-dip strategy)
            if 'dip_pct' in display_df.columns:
                display_columns.append('dip_pct')
            
            st.dataframe(
                display_df[display_columns],
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv,
                file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Trade distribution
            st.markdown("---")
            st.header("📊 Trade Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L distribution
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Histogram(
                    x=trades_df['pnl_pct'],
                    nbinsx=30,
                    name='P&L Distribution',
                    marker_color='#1f77b4'
                ))
                fig_pnl.update_layout(
                    title='P&L Distribution (%)',
                    xaxis_title='P&L (%)',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Trades by ticker
                ticker_counts = trades_df['ticker'].value_counts()
                fig_ticker = go.Figure()
                fig_ticker.add_trace(go.Bar(
                    x=ticker_counts.index,
                    y=ticker_counts.values,
                    marker_color='#2ca02c'
                ))
                fig_ticker.update_layout(
                    title='Trades by Ticker',
                    xaxis_title='Ticker',
                    yaxis_title='Number of Trades',
                    height=400
                )
                st.plotly_chart(fig_ticker, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred during backtesting: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Strategy Simulator MVP | Built with Streamlit & Alpaca Markets API</p>
    </div>
    """,
    unsafe_allow_html=True
)
