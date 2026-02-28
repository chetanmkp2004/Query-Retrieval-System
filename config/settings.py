"""Configuration helpers for the LLM ML agent."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # Change from "openai/gpt-4o-mini"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_RANDOM_STATE = 42


@dataclass
class AppSettings:
    model_name: str = DEFAULT_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    random_state: int = DEFAULT_RANDOM_STATE
    openrouter_base_url: str = OPENROUTER_BASE_URL


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_openrouter_api_key(api_key: str) -> None:
    os.environ["OPENAI_API_KEY"] = api_key


def build_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI client configured for OpenRouter."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=OPENROUTER_BASE_URL,
    )
