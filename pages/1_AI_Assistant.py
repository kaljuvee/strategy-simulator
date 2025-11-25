"""
AI Assistant Page
Use XAI Grok to develop and analyze trading strategies
"""

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Strategy Assistant")
st.markdown("Use AI to develop, analyze, and optimize trading strategies")

# Initialize XAI client
XAI_API_KEY = os.getenv('XAI_API_KEY')

if not XAI_API_KEY:
    st.error("❌ XAI_API_KEY not found in environment variables. Please set it in .env file.")
    st.stop()

# Initialize OpenAI client with XAI endpoint
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar with sample strategies
st.sidebar.header("📚 Sample Strategies")
st.sidebar.markdown("Click a button to analyze a common trading strategy:")

# Sample strategy buttons
sample_strategies = {
    "Moving Average Crossover": """
    Analyze the Moving Average Crossover strategy:
    - Buy when short-term MA crosses above long-term MA (golden cross)
    - Sell when short-term MA crosses below long-term MA (death cross)
    - Common parameters: 50-day and 200-day MAs
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "RSI Oversold/Overbought": """
    Analyze the RSI (Relative Strength Index) strategy:
    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    - 14-period RSI is standard
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "Bollinger Bands Bounce": """
    Analyze the Bollinger Bands strategy:
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Standard: 20-period MA with 2 standard deviations
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "MACD Signal": """
    Analyze the MACD (Moving Average Convergence Divergence) strategy:
    - Buy when MACD line crosses above signal line
    - Sell when MACD line crosses below signal line
    - Standard: 12, 26, 9 periods
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "Mean Reversion": """
    Analyze the Mean Reversion strategy:
    - Buy when price deviates significantly below mean
    - Sell when price returns to or above mean
    - Can use z-score or percentage deviation
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "Momentum Trading": """
    Analyze the Momentum Trading strategy:
    - Buy stocks with strong recent performance
    - Hold for short to medium term
    - Can use rate of change (ROC) or relative strength
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "Breakout Strategy": """
    Analyze the Breakout Trading strategy:
    - Buy when price breaks above resistance level
    - Sell when price breaks below support level
    - Can use volume confirmation
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """,
    
    "Pairs Trading": """
    Analyze the Pairs Trading (statistical arbitrage) strategy:
    - Identify correlated stock pairs
    - Buy underperformer, short outperformer when spread widens
    - Close positions when spread normalizes
    
    Provide: strategy explanation, pros/cons, best market conditions, and Python implementation example.
    """
}

for strategy_name, strategy_prompt in sample_strategies.items():
    if st.sidebar.button(strategy_name, use_container_width=True):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": strategy_prompt
        })
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Ask about strategy performance metrics
- Request backtesting code examples
- Compare multiple strategies
- Get risk management advice
- Ask for strategy optimization ideas
""")

# Main chat interface
st.markdown("---")
st.subheader("💬 Chat with AI Strategy Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about trading strategies, backtesting, or get code examples..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call XAI Grok API
            stream = client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert trading strategy advisor and quantitative analyst. 
                        You help users develop, analyze, and optimize trading strategies.
                        
                        When discussing strategies:
                        - Explain the logic clearly
                        - Provide pros and cons
                        - Suggest best market conditions
                        - Include Python code examples when relevant
                        - Mention key performance metrics to track
                        - Provide risk management considerations
                        
                        Be concise but thorough. Use markdown formatting for code blocks.
                        Focus on practical, implementable advice."""
                    }
                ] + st.session_state.messages,
                stream=True,
                temperature=0.7
            )
            
            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"❌ Error calling XAI API: {str(e)}"
            st.error(error_message)
            full_response = error_message
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if len(st.session_state.messages) > 0:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Information section
st.markdown("---")
st.markdown("""
### 🎯 What Can the AI Assistant Help With?

**Strategy Development:**
- Explain popular trading strategies
- Suggest strategies for different market conditions
- Help design custom strategies based on your ideas

**Code & Implementation:**
- Provide Python code examples for strategies
- Show how to calculate technical indicators
- Demonstrate backtesting implementations

**Analysis & Optimization:**
- Analyze strategy performance metrics
- Suggest improvements and optimizations
- Compare multiple strategies
- Provide risk management recommendations

**Education:**
- Explain technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Teach backtesting best practices
- Discuss position sizing and risk management
- Explain market conditions and when to use different strategies

### 📊 Example Questions:

- "How does the RSI indicator work and how can I use it for trading?"
- "Show me Python code to implement a moving average crossover strategy"
- "What are the best strategies for volatile markets?"
- "How do I calculate the Sharpe ratio for my strategy?"
- "Compare momentum trading vs mean reversion strategies"
- "What risk management techniques should I use?"
- "How can I optimize my buy-the-dip strategy parameters?"
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>AI Strategy Assistant | Powered by XAI Grok</p>
    </div>
    """,
    unsafe_allow_html=True
)
