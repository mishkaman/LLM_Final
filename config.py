import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()



OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

SUPPORTS_TEMPERATURE: bool = os.getenv("SUPPORTS_TEMPERATURE", "true").lower() == "true"


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
PROBLEMS_PATH: str = os.path.join(DATA_DIR, "problems.json")
DEBATE_RESULTS_PATH: str = os.path.join(RESULTS_DIR, "debate_results.json")
METRICS_PATH: str = os.path.join(RESULTS_DIR, "metrics.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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

AGENT_ORDER = ["Alpha", "Beta", "Gamma", "Delta"]
