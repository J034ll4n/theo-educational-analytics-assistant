@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo py launcher failed, trying python...
    python -m venv .venv
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist "data\dados.parquet" (
  echo Running ETL: relatorio.csv -^> dados.parquet
  python scripts\etl.py
)

if not exist "models\modelo.joblib" (
  echo Training ML model...
  python scripts\train_model.py
)

if exist "models\modelo.joblib" (
  echo Refreshing data\dados.parquet with ML column risco...
  python scripts\etl.py
)

echo Starting Streamlit...
set "PYTHONPATH=%~dp0"
python -m streamlit run app\main.py --server.headless true
pause
