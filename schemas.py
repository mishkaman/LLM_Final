"""Pydantic structured-output models for the debate.

Why structured outputs: the OpenAI Responses API can be handed a Pydantic class
via ``text_format=`` and will constrain the model to return JSON matching it.
That pushes all the "did the model answer in a parseable shape?" risk into the
API boundary, so the orchestration code never has to defensively parse text.

Starting with just the Stage 0 (role) and Stage 1 (solve) contracts; the review,
refinement, and judgment models get added as those stages come online.
"""
from typing import List  # noqa: F401  (used as later stages are added)

from pydantic import BaseModel, Field


# Stage 0: role self-assessment
class ConfidenceByRole(BaseModel):
    """An agent's self-rated fit for each role, each in [0, 1]."""

    solver: float = Field(description="Confidence this agent would make a good solver.")
    judge: float = Field(description="Confidence this agent would make a good judge.")


class RoleSelfAssessment(BaseModel):
    """Stage 0 output: how an agent rates itself for this specific problem."""

    preferred_role: str = Field(description="Either 'solver' or 'judge'.")
    confidence: ConfidenceByRole
    reasoning: str = Field(description="Brief justification for the preference.")


# Stage 1: independent solution
class SolverSolution(BaseModel):
    """A single solver's complete, independent attempt."""

    reasoning: str = Field(description="Full step-by-step working.")
    final_answer: str = Field(description="Concise answer only, no extra words.")
    confidence: float = Field(description="Self-rated confidence in [0, 1].")
