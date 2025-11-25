import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the alpaca agent
try:
    from utils.alpaca_agent import get_response
    agent_available = True
except ImportError as e:
    agent_available = False
    import_error = str(e)

# Page configuration
st.set_page_config(
    page_title="Alpaca Chat",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Alpaca Chat Assistant")
st.markdown("Chat naturally with your Alpaca trading account using AI")

# Check if agent is available
if not agent_available:
    st.error(f"⚠️ Alpaca Agent not available. Please install required packages: {import_error}")
    st.info("""
    Required packages:
    - langchain-openai
    - langgraph
    - alpaca-py
    
    Install with: `pip install langchain-openai langgraph alpaca-py`
    """)
    st.stop()

# Check for required API keys
if not os.getenv("XAI_API_KEY"):
    st.error("⚠️ XAI_API_KEY not found in environment variables")
    st.stop()

if not os.getenv("ALPACA_PAPER_API_KEY") or not os.getenv("ALPACA_PAPER_SECRET_KEY"):
    st.error("⚠️ Alpaca API keys not found in environment variables")
    st.stop()

# Initialize session state for chat history
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "alpaca_chat_session"

# Sidebar with sample questions
st.sidebar.header("💡 Sample Questions")
st.sidebar.markdown("Click a button to ask a common question:")

sample_questions = {
    "What are my positions?": "Show me all my current positions with their P/L",
    "What's my account status?": "What's my current account balance and buying power?",
    "Place order for MSFT": "Buy 10 shares of MSFT at market price",
    "Show recent orders": "What are my recent orders?",
    "Available assets": "Show me some available US equity assets for trading",
    "Close AAPL position": "Sell all my AAPL shares"
}

for question_label, question_text in sample_questions.items():
    if st.sidebar.button(question_label, use_container_width=True):
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": question_text
        })
        
        # Get response from agent
        with st.spinner("🤔 Thinking..."):
            try:
                response = get_response(question_text, st.session_state.thread_id)
                
                # Extract the assistant's response
                assistant_message = None
                for msg in reversed(response["messages"]):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        assistant_message = msg.content
                        break
                
                if assistant_message:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
            except Exception as e:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })
        
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
This chat interface uses AI to interact with your Alpaca trading account.

**Capabilities:**
- Check account status
- View positions and P/L
- Place market orders
- View order history
- Search for assets

**Note:** This uses paper trading for safety.
""")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_messages = []
    st.rerun()

# Main chat interface
st.markdown("---")

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your account, positions, or place orders..."):
    # Add user message to chat
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = get_response(prompt, st.session_state.thread_id)
                
                # Extract the assistant's response
                assistant_message = None
                for msg in reversed(response["messages"]):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        assistant_message = msg.content
                        break
                
                if assistant_message:
                    st.markdown(assistant_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                else:
                    error_msg = "Sorry, I couldn't generate a response. Please try again."
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Alpaca Chat Assistant | Powered by LangGraph & XAI Grok")
