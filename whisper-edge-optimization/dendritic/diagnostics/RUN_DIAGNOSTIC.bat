@echo off
echo ================================================================
echo GPU and Training Stack Diagnostic Test
echo ================================================================
echo.
echo This will test each component step-by-step:
echo - PyTorch and CUDA
echo - GPU memory operations
echo - Whisper model loading
echo - PerforatedAI (if installed)
echo.
echo Watch for the last successful step if a crash occurs.
echo.
echo ================================================================
echo.

cd /d "%~dp0"

python test_diagnostic.py 2>&1

echo.
echo ================================================================
echo Diagnostic complete.
echo ================================================================
echo.

REM Auto-exit after 10 seconds
timeout /t 10 /nobreak > nul
exit /b %ERRORLEVEL%
