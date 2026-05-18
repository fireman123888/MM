"""Runtime configuration for the legal RAG MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLM_BASE_URL = "http://127.0.0.1:49328/v1"
DEFAULT_LLM_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout_seconds: float = 30.0
    use_llm: bool = True
    retrieval_top_k: int = 6
    corpus_path: str = "data/lawrag_corpus.jsonl"


def load_settings() -> Settings:
    """Load settings from environment variables.

    The API key intentionally comes from the environment instead of a checked-in
    file. Supported names:
    - LAWRAG_LLM_API_KEY
    - OPENAI_API_KEY
    """

    load_env_file()
    api_key = os.getenv("LAWRAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    return Settings(
        llm_base_url=os.getenv("LAWRAG_LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/"),
        llm_api_key=api_key,
        llm_model=os.getenv("LAWRAG_LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_timeout_seconds=float(os.getenv("LAWRAG_LLM_TIMEOUT", "30")),
        use_llm=os.getenv("LAWRAG_USE_LLM", "1") not in {"0", "false", "False"},
        retrieval_top_k=int(os.getenv("LAWRAG_RETRIEVAL_TOP_K", "6")),
        corpus_path=os.getenv("LAWRAG_CORPUS_PATH", "data/lawrag_corpus.jsonl"),
    )


def load_env_file(path: str = ".env") -> None:
    """Load a simple KEY=VALUE env file without adding a dependency."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
