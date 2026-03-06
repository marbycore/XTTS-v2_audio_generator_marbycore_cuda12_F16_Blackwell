#!/bin/bash
set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "   MARBYCORE - XTTS-v2 AUDIO GENERATOR (LINUX / macOS)"
echo "============================================================"
echo ""

# 1. SEARCH FOR SYSTEM PYTHON (MUST BE 3.10 FOR COQUI)
if command -v python3.10 &>/dev/null; then
    PYTHON_EXE="python3.10"
elif command -v python3 &>/dev/null && python3 --version 2>&1 | grep -q '3\.10'; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null && python --version 2>&1 | grep -q '3\.10'; then
    PYTHON_EXE="python"
else
    echo "[ERROR] Python 3.10 is required for Coqui XTTS-v2!"
    echo "We found a different Python version on your system."
    echo "Please install Python 3.10 and ensure it is in PATH."
    exit 1
fi

# 2. AUTOMATIC ENVIRONMENT SETUP
if [ ! -f ".venv/bin/activate" ] || [ ! -f ".venv/.installed_v1" ]; then
    
    if [ ! -d ".venv" ]; then
        echo "[SETUP] Creating isolated environment using $PYTHON_EXE..."
        $PYTHON_EXE -m venv .venv || { echo "[ERROR] Failed to create virtual environment."; exit 1; }
    fi
    
    echo "[SETUP] Installing technical dependencies (this may take a few minutes)..."
    source .venv/bin/activate
    python -m pip install --upgrade pip

    # For Linux Nvidia, install nightly PyTorch (M1/M2 macOS will ignore cu128 automatically)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "[SETUP] Installing optimal PyTorch version for Linux..."
        python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 || true
    fi

    echo "[SETUP] Installing Coqui XTTS and remaining dependencies..."
    python -m pip install -r requirements.txt || { echo "[ERROR] Failed to install dependencies."; exit 1; }
    
    # Create the flag file to signal successful installation
    echo "ALREADY_INSTALLED" > ".venv/.installed_v1"
    echo "[SETUP] Installation complete!"
else
    echo "[1/3] Activating environment..."
    source .venv/bin/activate
fi

# 3. VERIFY SCRIPT
if [ ! -f "app.py" ]; then
    echo "[ERROR] Critical file missing: app.py"
    exit 1
fi

# 4. START ENGINE
echo "[2/3] Verifying environment..."
echo "[3/3] Launching Engine..."
echo ""
echo "------------------------------------------------------------"
echo "The dashboard will open automatically in your browser."
echo "Keep this window open."
echo "------------------------------------------------------------"
echo ""

python app.py || {
    echo ""
    echo "[ERROR] The engine stopped unexpectedly."
    exit 1
}

exit 0
