"""
Demo 13: Streamlit Integration with FastAPI Streaming

This project demonstrates a Streamlit chatbot that integrates with the FastAPI backend
from demo-08-llm-stream-endpoint using the /query/stream endpoint for real-time streaming responses.
"""

import streamlit as st
import requests
import uuid
import json

# -------------------------------
# Configuration
# -------------------------------
API_BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE_URL}/query/stream"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Streaming Chat",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ AI Streaming Chat Assistant")
st.caption("Real-time streaming responses • Powered by FastAPI + Streamlit")

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# -------------------------------
# Sidebar - API Status & Info
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Info
    st.info(f"**Streaming Endpoint:**\n`{STREAM_ENDPOINT}`")
    
    st.divider()
    
    st.markdown("""
    ### 📝 Instructions
    1. Start demo-08 FastAPI server:
       ```bash
       cd demo-08-llm-stream-endpoint
       uv run fastapi dev main.py
       ```
    2. Ask your question below
    3. Watch the response stream in real-time!
    """)
    
    st.divider()
    
    st.markdown("""
    ### ✨ Streaming Features
    - Real-time token-by-token display
    - Server-Sent Events (SSE)
    - ChatGPT-like typing effect
    - Instant response start
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------
# Display previous messages
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Helper function to parse SSE stream
# -------------------------------
def parse_sse_stream(response):
    """
    Parse Server-Sent Events stream from the API response.
    
    Args:
        response: requests.Response object with streaming enabled
        
    Yields:
        str: Individual tokens/chunks from the stream
    """
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            # SSE format: "data: <content>"
            if decoded_line.startswith('data: '):
                content = decoded_line[6:]  # Remove "data: " prefix
                if content and content != "[DONE]":
                    yield content

# -------------------------------
# Chat Input and Streaming Response
# -------------------------------
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Make streaming API request to FastAPI backend
            with requests.post(
                STREAM_ENDPOINT,
                json={"prompt": prompt},
                stream=True,  # Enable streaming
                timeout=60
            ) as response:
                
                if response.status_code == 200:
                    # Stream the response token by token
                    for chunk in parse_sse_stream(response):
                        full_response += chunk
                        # Update the message placeholder with accumulated text
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Save complete response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                else:
                    error_msg = f"❌ Error: API returned status code {response.status_code}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to API. Please ensure the FastAPI server is running at http://localhost:8000"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
            
        except requests.exceptions.Timeout:
            error_msg = "⏱️ Request timed out. Please try again."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("💡 Tip: This chatbot uses Server-Sent Events (SSE) for real-time streaming from demo-08!")
