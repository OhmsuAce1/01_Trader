@echo off
title Webull US Stock Pro Trader
cd /d "%~dp0"
echo ========================================================
echo Starting Webull US Stock Pro Trader on Localhost...
echo ========================================================
.venv\Scripts\streamlit run app.py
pause
