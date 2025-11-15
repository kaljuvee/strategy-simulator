"""
VIX Fear Index Strategy Page
Trade based on market volatility - buy when fear is high
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('/home/ubuntu/strategy-simulator')
from utils.backtester_util import backtest_vix_strategy

# Page configuration
st.set_page_config(
    page_title="VIX Strategy",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ VIX Fear Index Strategy")
st.markdown("Trade based on market volatility - buy when fear is high (VIX > threshold)")

# Sidebar configuration
st.sidebar.header("VIX Strategy Configuration")

st.sidebar.markdown("---")
st.sidebar.subheader("Stock Selection")

# Stock options (same as Home page)
MAG7_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Meta": "META"
}

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
    help="Select stocks or ETFs to trade when VIX is high"
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
st.sidebar.subheader("VIX Strategy Parameters")

# VIX threshold
vix_threshold = st.sidebar.slider(
    "VIX Threshold",
    min_value=10.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
    help="Buy when VIX exceeds this level (higher = more fear)"
)

# Hold period
hold_overnight = st.sidebar.radio(
    "Hold Period",
    options=[True, False],
    format_func=lambda x: "Hold Overnight" if x else "Sell Same Day",
    help="Whether to hold positions overnight or sell same day"
)

# Info box about VIX
st.info("""
**About VIX (Fear Index):**
- VIX measures expected market volatility over the next 30 days
- **Low VIX (< 15)**: Market is calm, low fear
- **Normal VIX (15-20)**: Moderate volatility
- **High VIX (20-30)**: Elevated fear, increased volatility
- **Very High VIX (> 30)**: Extreme fear, panic selling

**Strategy Logic:** Buy stocks when VIX is high (fear is high), as markets often rebound after panic.
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
if st.button("🚀 Run VIX Strategy Backtest", type="primary", use_container_width=True):
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        st.stop()
    
    with st.spinner("Running VIX strategy backtest... This may take a moment."):
        try:
            # Run backtest
            results = backtest_vix_strategy(
                symbols=selected_symbols,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                initial_capital=initial_capital,
                position_size=position_size / 100,
                vix_threshold=vix_threshold,
                hold_overnight=hold_overnight
            )
            
            if results is None:
                st.error("❌ No trades were generated during the backtest period. Try adjusting parameters or date range.")
                st.stop()
            
            trades_df, metrics = results
            
            # Display results
            st.success("✅ VIX strategy backtest completed successfully!")
            
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
                title='Portfolio Value Over Time (VIX Strategy)',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # VIX levels during trades
            st.markdown("---")
            st.header("⚡ VIX Levels During Trades")
            
            fig_vix = go.Figure()
            
            fig_vix.add_trace(go.Scatter(
                x=trades_df['entry_time'],
                y=trades_df['vix_level'],
                mode='markers',
                name='VIX at Trade Entry',
                marker=dict(
                    size=8,
                    color=trades_df['vix_level'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="VIX Level")
                )
            ))
            
            # Add threshold line
            fig_vix.add_hline(
                y=vix_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VIX Threshold ({vix_threshold})",
                annotation_position="right"
            )
            
            fig_vix.update_layout(
                title='VIX Levels When Trades Were Entered',
                xaxis_title='Date',
                yaxis_title='VIX Level',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_vix, use_container_width=True)
            
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
            display_df['vix_level'] = display_df['vix_level'].apply(lambda x: f"{x:.2f}")
            
            # Select columns to display
            display_columns = [
                'entry_time', 'exit_time', 'ticker', 'shares', 
                'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                'capital_after', 'vix_level'
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
                file_name=f"vix_strategy_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
            
            # VIX statistics
            st.markdown("---")
            st.header("📈 VIX Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average VIX at Entry", f"{trades_df['vix_level'].mean():.2f}")
            with col2:
                st.metric("Max VIX at Entry", f"{trades_df['vix_level'].max():.2f}")
            with col3:
                st.metric("Min VIX at Entry", f"{trades_df['vix_level'].min():.2f}")
            with col4:
                st.metric("VIX Std Dev", f"{trades_df['vix_level'].std():.2f}")
            
        except Exception as e:
            st.error(f"❌ An error occurred during backtesting: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>VIX Fear Index Strategy | Built with Streamlit & Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True
)
