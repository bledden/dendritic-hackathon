@echo off
REM Windows Setup Script for Dendritic Whisper Compression
REM ========================================================

echo ========================================================================
echo   Dendritic Whisper - Windows Setup Script
echo ========================================================================
echo.

REM Check Python version
echo [1/8] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

python --version

REM Create virtual environment
echo.
echo [2/8] Creating Python virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install PyTorch with CUDA support
echo.
echo [4/8] Installing PyTorch with CUDA 12.4 support...
echo This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install project dependencies
echo.
echo [5/8] Installing project dependencies...
pip install openai-whisper datasets transformers soundfile librosa tqdm jiwer pandas scikit-learn wandb python-dotenv

REM Install PerforatedAI
echo.
echo [6/8] Installing PerforatedAI library...
if exist PerforatedAI (
    cd PerforatedAI
    if not exist setup.py (
        echo PerforatedAI directory is empty, cloning repository...
        git clone https://github.com/PerforatedAI/PerforatedAI.git .
    )
    pip install -e .
    cd ..
    echo PerforatedAI installed successfully
) else (
    echo ERROR: PerforatedAI directory not found
    pause
    exit /b 1
)

REM Create .env file
echo.
echo [7/8] Setting up environment variables...
if not exist .env (
    echo Creating .env file...
    echo # PerforatedAI API Credentials > .env
    echo PAIEMAIL=hacker_token@perforatedai.com >> .env
    echo PAITOKEN=MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg== >> .env
    echo. >> .env
    echo # WandB API Key (optional) >> .env
    echo # WANDB_API_KEY=your_key_here >> .env
    echo .env file created
) else (
    echo .env file already exists
)

REM Verify installation
echo.
echo [8/8] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
python -c "import whisper; print('Whisper: OK')"
python -c "from perforatedai import utils_perforatedai as UPA; print('PerforatedAI: OK')"

echo.
echo ========================================================================
echo   Setup Complete!
echo ========================================================================
echo.
echo IMPORTANT NOTE FOR RTX 5090 USERS:
echo Your RTX 5090 GPU has CUDA capability sm_120 (Blackwell architecture).
echo The current stable PyTorch (2.6.0) does not fully support this architecture.
echo.
echo OPTIONS:
echo 1. Use CPU mode for development/testing (slower but works)
echo 2. Wait for PyTorch nightly builds with sm_120 support
echo 3. Use WSL2 with ROCm (if available)
echo.
echo To run training on CPU, add --device cpu to training commands.
echo.
echo Next steps:
echo 1. Activate environment: venv\Scripts\activate.bat
echo 2. Navigate to training: cd whisper-edge-optimization\dendritic
echo 3. Run test (CPU mode): python train_dendritic_full.py --device cpu --val-max-samples 10
echo.
echo See WINDOWS_SETUP.md for detailed documentation
echo ========================================================================
pause
