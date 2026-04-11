"""Cliente Ollama via LangChain."""

from __future__ import annotations

from typing import Any

import requests
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from passos_magico.llm.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2)
        return r.status_code == 200
    except (OSError, requests.RequestException):
        return False


def get_chat_ollama(**kwargs: Any) -> ChatOllama:
    opts = {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    opts.update(kwargs)
    return ChatOllama(**opts)


def invoke_string(system: str, user: str, temperature: float = 0.1) -> str:
    chat = get_chat_ollama(temperature=temperature)
    messages: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    out = chat.invoke(messages)
    if isinstance(out, AIMessage):
        return str(out.content)
    return str(out)


def stream_tokens(system: str, user: str, temperature: float = 0.2):
    chat = get_chat_ollama(temperature=temperature)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    for chunk in chat.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content
