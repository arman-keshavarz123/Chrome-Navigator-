# Chrome Navigator

A voice-controlled Chrome extension that allows users to control browser actions using natural language commands through a locally-deployed machine learning model.

## Features

- **Voice Control**: Use natural language to control your browser
- **Local Processing**: All speech recognition and language processing happens on your machine
- **Comprehensive Coverage**: Supports common browsing tasks like:
  - Navigation (go back, go forward, navigate to URLs)
  - Tab management (open, close, switch tabs)
  - Form interaction (click buttons, type text)
  - Scrolling and page manipulation
  - Search functionality
  - Page refresh and reload

## How It Works

1. **Voice Input**: Click the extension icon and speak your command
2. **Local Processing**: Your voice is converted to text and sent to a local Flask server
3. **ML Inference**: A trained transformer model converts natural language to RPC commands
4. **Browser Execution**: The extension executes the command using Chrome's APIs

## Installation & Setup


### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Chrome-Navigator.git
cd Chrome-Navigator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Local Server

```bash
python llm_server_t5.py
```

### 4. Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked" and select the `extension/` folder
4. The Chrome Navigator extension should now appear in your toolbar

## Usage

1. **Activate Voice Control**: Click the Chrome Navigator extension icon in your toolbar
2. **Speak Your Command**: Say something like "scroll down by 300 pixels" or "click the login button"
3. **Watch It Execute**: The extension will automatically perform the requested action

### Example Commands

- "open a new tab"
- "scroll down by 200 pixels"
- "click the sign in button"
- "go back two steps"
- "search up netflix"

