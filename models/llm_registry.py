"""LLM Registry - Centralized LLM initialization with environment configuration."""

from __future__ import annotations
import os
from typing import Optional
import httpx
from langchain_openai import ChatOpenAI


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback."""
    return os.getenv(name, default)


def init_llm(name: Optional[str] = None) -> ChatOpenAI:
    """Initialize and return a ChatOpenAI instance by logical name.

    Supported names:
      - gpt-4o-mini
      - gpt-oss-120b
      - gpt-oss-20b
      - Llama33
      - gauss2-3-37b
    """
    # allow override via env
    name = name or _env("LLM_NAME")

    # common knobs
    max_tokens = 1024

    if name == "gpt-4o-mini":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=_env("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=max_tokens,
        )

    elif name in {"gpt-oss-120b", "gpt-oss-20b"}:
        # pick base_url per model from env
        base_url = (
            _env("GPT_OSS_BASE_URL_120B")
            if name == "gpt-oss-120b"
            else _env("GPT_OSS_BASE_URL_20B")
        )
        api_key = _env("GPT_OSS_API_KEY")
        x_api_key = _env("GPT_OSS_X_API_KEY")

        # effort: low for 120b, high for 20b
        effort = "low" if name == "gpt-oss-120b" else "high"

        default_headers = {"X-API-KEY": x_api_key} if x_api_key else None

        return ChatOpenAI(
            model="llm",
            temperature=0.0,
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            extra_body={
                "skip_special_tokens": False,
                "reasoning": {"effort": effort},
            },
            max_tokens=max_tokens,
        )

    elif name == "Llama33":
        # Get config from env
        base_url = _env("LLAMA_BASE_URL")
        api_key = _env("LLAMA_API_KEY")
        x_api_key = _env("LLAMA_X_API_KEY")

        default_headers = {"X-API-KEY": x_api_key} if x_api_key else None

        return ChatOpenAI(
            model="Llama-3.3-70B-Instruct",
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"top_k": 50},
        )

    elif name == "gauss2-3-37b":
        base_url = _env("GAUSS_BASE_URL")
        api_key = _env("GAUSS_API_KEY")
        x_api_key = _env("GAUSS_X_API_KEY")
        verify_ssl = (_env("GAUSS_VERIFY_SSL") or "true").lower() == "true"

        http_client = httpx.Client(verify=verify_ssl, timeout=60.0, http2=False)

        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"X-API-KEY": x_api_key} if x_api_key else None,
            model=_env("GAUSS_MODEL_ID"),
            temperature=0.0,
            http_client=http_client,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM name: {name}")