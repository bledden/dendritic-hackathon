@echo off
echo ================================================================
echo Full Training Stack Diagnostic (including dataset)
echo ================================================================
echo.
echo This will test:
echo - PyTorch and CUDA
echo - Whisper model
echo - LibriSpeech dataset loading
echo - DataLoader operations
echo - Forward passes with BF16
echo.
echo This is the most comprehensive test before actual training.
echo.
echo ================================================================
echo.

cd /d "%~dp0"

python test_diagnostic_full.py 2>&1

echo.
echo ================================================================
echo Full diagnostic complete.
echo ================================================================
echo.

REM Auto-exit after 10 seconds
timeout /t 10 /nobreak > nul
exit /b %ERRORLEVEL%
