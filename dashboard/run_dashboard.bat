@echo off
cd /d "%~dp0"
echo Installing packages (first run only takes a minute)...
python -m pip install -q -r requirements.txt
python -m streamlit run app.py
pause
