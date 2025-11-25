"""
Strategy Simulator - Home Page
Main backtesting interface for Buy-The-Dip strategy
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.backtester_util import backtest_buy_the_dip
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

st.title("📈 Strategy Simulator - Buy The Dip")
st.markdown("Backtest and analyze trading strategies with real market data")

# Sidebar configuration
st.sidebar.header("Strategy Configuration")

# Strategy selection (currently only buy-the-dip)
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["buy-the-dip"],
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
            # Run backtest
            results = backtest_buy_the_dip(
                symbols=selected_symbols,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                initial_capital=initial_capital,
                position_size=position_size / 100,
                dip_threshold=dip_threshold / 100,
                hold_days=hold_days,
                take_profit=take_profit / 100,
                stop_loss=stop_loss / 100
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
            
            st.markdown("---")
            st.header("📈 Equity Curve")
            
            # Create equity curve chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades_df['entry_time'],
                y=trades_df['capital_after'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
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
            display_df['dip_pct'] = display_df['dip_pct'].apply(lambda x: f"{x:.2f}%")
            
            # Select columns to display
            display_columns = [
                'entry_time', 'exit_time', 'ticker', 'shares', 
                'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                'hit_target', 'hit_stop', 'capital_after', 'dip_pct'
            ]
            
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
