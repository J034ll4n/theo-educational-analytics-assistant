@echo off
REM Um duplo clique (ou a tarefa de build no VS Code/Cursor) instala dependencias e abre o Streamlit.
REM O Windows nao permite icone proprio em ficheiros .bat. Para ver a imagem no atalho,
REM execute criar_atalho.ps1 e use "Passos Mágicos.lnk" (ou arraste o .lnk para o Ambiente de Trabalho).
setlocal
cd /d "%~dp0"

echo [Passos Magicos] Pasta do projeto: %CD%

if not exist ".venv\Scripts\python.exe" (
  echo [Passos Magicos] A criar ambiente virtual .venv...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo py launcher falhou, a tentar python...
    python -m venv .venv
  )
)

call ".venv\Scripts\activate.bat"
echo [Passos Magicos] A atualizar pip e a instalar requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist "data\dados.parquet" (
  echo Running ETL: relatorio.csv -^> dados.parquet
  python scripts\etl.py
) else (
  if exist "modelo_risco_aluno.pkl" (
    echo Refreshing data\dados.parquet with ML column risco...
    python scripts\etl.py
  )
)

echo [Passos Magicos] A iniciar Streamlit — deve abrir o browser; o URL tambem aparece no terminal.
set "PYTHONPATH=%~dp0"
python -m streamlit run app\main.py
pause
