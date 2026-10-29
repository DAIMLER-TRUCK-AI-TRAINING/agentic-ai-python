"""
Demo 12: Streamlit Integration with FastAPI

This project demonstrates a Streamlit chatbot that integrates with the FastAPI backend
from demo-04-console-llm-app-to-rest-api using the /query endpoint.
"""

import streamlit as st
import requests
import uuid

# -------------------------------
# Configuration
# -------------------------------
API_BASE_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chat Assistant")
st.caption("Powered by FastAPI + Streamlit")

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# -------------------------------
# Sidebar - API Status Check
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check API health
    if st.button("Check API Status"):
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ API Status: {data.get('status', 'unknown')}")
                st.info(f"Service: {data.get('service', 'N/A')}")
                st.info(f"Version: {data.get('version', 'N/A')}")
            else:
                st.error(f"❌ API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.markdown("""
    ### 📝 Instructions
    1. Make sure demo-04 FastAPI server is running:
       ```bash
       cd demo-04-console-llm-app-to-rest-api
       uv run fastapi dev main.py
       ```
    2. Ask your question in the chat below
    3. Get AI-powered responses!
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
# Chat Input and Response
# -------------------------------
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Make API request to FastAPI backend
                response = requests.post(
                    QUERY_ENDPOINT,
                    json={"prompt": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response received")
                    model_used = data.get("model", "unknown")
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Show model info in a subtle way
                    st.caption(f"Model: {model_used}")
                    
                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    error_msg = f"Error: API returned status code {response.status_code}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to API. Please ensure the FastAPI server is running at http://localhost:8000"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("💡 Tip: This chatbot connects to demo-04 FastAPI backend. Make sure it's running!")
