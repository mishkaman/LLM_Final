"""Thin, persona-aware wrapper around the OpenAI Responses API.

Every stage of the debate calls ``structured_call(...)`` with a Pydantic model
as the response format. Personas differ only by system prompt + temperature,
which is how one model plays four distinct "agents".

First cut: a single plain call, no retry logic yet. Robustness (backoff on
transient errors, truncation recovery) is added later once real runs surface
those failure modes.
"""
from typing import Optional, Type, TypeVar

from pydantic import BaseModel
from openai import OpenAI

import config

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazily create a single shared OpenAI client."""
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def structured_call(
    system_prompt: str,
    user_prompt: str,
    response_format: Type[T],
    persona: Optional[str] = None,
    temperature: Optional[float] = None,
) -> T:
    """Make one structured (schema-constrained) call and return the parsed object.

    Args:
        system_prompt: the role/persona system instruction.
        user_prompt: the concrete task (problem, solutions to review, etc.).
        response_format: a Pydantic model class the reply must conform to.
        persona: optional persona name; supplies a default temperature.
        temperature: explicit override; wins over the persona default.

    Returns:
        An instance of ``response_format``.
    """
    client = get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": config.MODEL_NAME,
        "input": messages,
        "text_format": response_format,
    }

    temp = temperature
    if temp is None and persona is not None:
        temp = config.PERSONAS[persona]["temperature"]
    if config.SUPPORTS_TEMPERATURE and temp is not None:
        kwargs["temperature"] = temp

    response = client.responses.parse(**kwargs)
    return response.output_parsed
