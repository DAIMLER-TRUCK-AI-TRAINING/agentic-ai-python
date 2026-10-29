# Streamlit Echo Agent

A simple interactive web application built with Streamlit that creates an echo agent - it takes user input and responds by echoing back the same message with an "Echo:" prefix.

## Features

- **Interactive Chat Interface**: Clean, modern chat UI powered by Streamlit
- **Session Management**: Each user session gets a unique ID for tracking
- **Message History**: Maintains conversation history during the session
- **Real-time Responses**: Instant echo responses to user input
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.12 or higher
- pip or uv package manager

## Installation

### Using pip

1. Clone or download this repository
2. Navigate to the project directory:

   ```bash

   cd demo-05-first-interactive-chat-app-with-streamlit

   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Using uv (recommended)

If you have `uv` installed:

```bash
uv sync
```

## Usage

1. Run the application:

   ```bash
   streamlit run main.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Start chatting! Type any message in the input field and press Enter to see the echo response.

## How It Works

The application uses Streamlit's session state to:

- Store conversation history as a list of message dictionaries
- Generate unique session IDs for each user
- Display messages in a chat-like interface
- Process user input and generate echo responses

### Key Components

- **Session State Management**: Tracks messages and session ID across interactions
- **Chat Interface**: Uses `st.chat_message()` for proper message display
- **Input Handling**: Captures user input with `st.chat_input()`
- **Echo Logic**: Simple string formatting to create echo responses

## Project Structure

```
streamlit_echo_agent/
├── main.py    # Main application file
├── pyproject.toml           # Project configuration and dependencies
├── README.md               # This file
├── .python-version         # Python version specification
├── .gitignore             # Git ignore rules
└── .venv/                 # Virtual environment (if using uv)
```

## Dependencies

- **streamlit>=1.28.0**: Web application framework
- **uuid**: Built-in Python module for unique ID generation

## Development

This is a simple demonstration project perfect for:

- Learning Streamlit basics
- Understanding session state management
- Building chat interfaces
- Prototyping conversational applications
