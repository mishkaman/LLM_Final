
from typing import List  # noqa: F401  (used as later stages are added)

from pydantic import BaseModel, Field


class ConfidenceByRole(BaseModel):

    solver: float = Field(description="Confidence this agent would make a good solver.")
    judge: float = Field(description="Confidence this agent would make a good judge.")


class RoleSelfAssessment(BaseModel):

    preferred_role: str = Field(description="Either 'solver' or 'judge'.")
    confidence: ConfidenceByRole
    reasoning: str = Field(description="Brief justification for the preference.")


class SolverSolution(BaseModel):

    reasoning: str = Field(description="Full step-by-step working.")
    final_answer: str = Field(description="Concise answer only, no extra words.")
    confidence: float = Field(description="Self-rated confidence in [0, 1].")
