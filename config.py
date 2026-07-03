"""Central configuration for the Multi-LLM Collaborative Debate System.

Design notes
------------
Everything tunable lives in this one module so the rest of the codebase can stay
declarative: no other file reads os.getenv directly. Values are resolved once
at import time from the environment (.env via python-dotenv) and then treated
as read-only constants everywhere downstream.
"""
import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


# API / model
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# One model plays all four roles (four calls, different personas/params).
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Reasoning models (o-series, gpt-5*) reject the `temperature` parameter. When using one of those, set SUPPORTS_TEMPERATURE=false in .env so we omit it.
SUPPORTS_TEMPERATURE: bool = os.getenv("SUPPORTS_TEMPERATURE", "true").lower() == "true"


# Filesystem paths
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
PROBLEMS_PATH: str = os.path.join(DATA_DIR, "problems.json")
DEBATE_RESULTS_PATH: str = os.path.join(RESULTS_DIR, "debate_results.json")
METRICS_PATH: str = os.path.join(RESULTS_DIR, "metrics.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Personas
# Four agents with deliberately different temperaments. Diversity of reasoning style is the whole point: a rigorous checker and a creative explorer tend to fail on *different* problems, so their disagreement is informative.
PERSONAS: Dict[str, Dict[str, object]] = {
    "Alpha": {
        "style": "rigorous and methodical; writes every algebraic/logical step "
                 "explicitly and re-derives results before committing to them",
        "temperature": 0.2,
    },
    "Beta": {
        "style": "creative and exploratory; looks for unconventional approaches, "
                 "reframes the problem, and actively hunts for overlooked edge cases",
        "temperature": 0.8,
    },
    "Gamma": {
        "style": "skeptical verifier; assumes the intuitive answer may be a trap, "
                 "stress-tests claims with small cases and counterexamples",
        "temperature": 0.5,
    },
    "Delta": {
        "style": "balanced generalist; clear, concise, well-structured reasoning "
                 "that weighs multiple angles before concluding",
        "temperature": 0.4,
    },
}

# Fixed ordering used for deterministic tie-breaks in Stage 0.5 role assignment.
AGENT_ORDER = ["Alpha", "Beta", "Gamma", "Delta"]
