# QBH Music Intelligence Platform

A hybrid Query-by-Humming (QBH) and Audio Fingerprinting platform.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- MySQL/MariaDB (for Dejavu fingerprinting)
- FFmpeg (installed and in your system PATH)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd QBH_Project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration & Security 🔒
Never commit your secret keys to GitHub. Follow these steps:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your unique credentials:
   - **Spotify**: Get these from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
   - **Database**: Configure your local MySQL settings.

**Note**: `.env` is already in `.gitignore` to prevent accidental exposure of your secrets.

### 4. Running the Platform
```bash
# Start the backend server
python app.py
```
Default URL: `http://127.0.0.1:5000`

## 🛠️ Components
- **Engine A**: Hybrid QBH (pYIN/TorchCREPE + DTW + ASR Reranking)
- **Engine B**: Audio Fingerprinting (Dejavu MySQL Backend)
- **Spotify Integration**: Rich metadata and recommendations

## 📜 Security Measures
- **Externalized Secrets**: All API keys and DB credentials are managed via environment variables.
- **Git Hygiene**: `.gitignore` is configured to exclude local uploads, logs, and sensitive `.env` files.
- **Path Portability**: System paths (like FFmpeg) are configurable via environment variables to support different OS environments.
