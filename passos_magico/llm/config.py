"""Configuração do Ollama — ajuste modelo e contexto aqui."""

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Modelo local (ex.: llama3, mistral, llama3.2)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
# Contexto amplo para dicionário + resumo anual + perguntas complexas (ajuste via env se o modelo exigir menos RAM)
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# Tempo máximo por pedido HTTP ao Ollama (invoke/stream), em segundos
OLLAMA_REQUEST_TIMEOUT = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "120"))
