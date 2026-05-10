#!/usr/bin/env bash
# Clone → executar: instala .venv, requirements, ETL se preciso, inicia Streamlit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Criando ambiente virtual .venv..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  else
    python -m venv .venv
  fi
fi

# shellcheck source=/dev/null
source ".venv/bin/activate"

echo "Atualizando pip e instalando dependências (requirements.txt)..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f "data/dados.parquet" ]]; then
  echo "A correr ETL (CSV → Parquet)..."
  python scripts/etl.py
elif [[ -f "modelo_risco_aluno.pkl" ]]; then
  echo "A atualizar data/dados.parquet (coluna risco)..."
  python scripts/etl.py
fi

export PYTHONPATH="$ROOT"
echo "A iniciar Streamlit (abra o URL no browser)..."
exec python -m streamlit run app/main.py
