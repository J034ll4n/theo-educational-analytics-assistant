"""Cliente Ollama via LangChain.

LangChain só é importado quando se chama `get_chat_ollama` / `invoke_string` / `stream_tokens`,
para a app (ex.: dashboards) arrancar sem essas dependências no MVP / Streamlit Cloud.
"""

from __future__ import annotations

from typing import Any

import requests

from passos_magico.llm.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_REQUEST_TIMEOUT,
)


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2)
        return r.status_code == 200
    except (OSError, requests.RequestException):
        return False


def get_chat_ollama(**kwargs: Any) -> Any:
    from langchain_community.chat_models import ChatOllama

    opts = {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "num_ctx": OLLAMA_NUM_CTX,
        "timeout": OLLAMA_REQUEST_TIMEOUT,
    }
    opts.update(kwargs)
    return ChatOllama(**opts)


def invoke_string(system: str, user: str, temperature: float = 0.1) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    chat = get_chat_ollama(temperature=temperature)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    out = chat.invoke(messages)
    if isinstance(out, AIMessage):
        return str(out.content)
    return str(out)


def stream_tokens(system: str, user: str, temperature: float = 0.2):
    from langchain_core.messages import HumanMessage, SystemMessage

    chat = get_chat_ollama(temperature=temperature)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    for chunk in chat.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content
