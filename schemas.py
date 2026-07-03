from typing import List

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
class SolutionError(BaseModel):
    """One concrete defect a reviewer flags in a peer's solution."""

    location: str = Field(description="Where the error is, e.g. 'Step 5'.")
    error_type: str = Field(description="e.g. 'logical_error', 'arithmetic_error'.")
    description: str
    severity: str = Field(description="One of: 'minor', 'major', 'critical'.")


class PeerReview(BaseModel):
    """Stage 2 output: one solver's structured critique of one peer solution."""

    solution_id: str = Field(description="Which solution this review targets.")
    strengths: List[str]
    weaknesses: List[str]
    errors: List[SolutionError]
    suggested_changes: List[str]
    overall_assessment: str = Field(
        description="One of: 'correct', 'promising_but_flawed', 'likely_incorrect'."
    )


class ChangeMade(BaseModel):
    """A single critique and how the solver responded to it."""

    critique: str
    response: str
    accepted: bool = Field(description="True if the solver accepted the critique.")


class RefinedSolution(BaseModel):
    """Stage 3 output: a solver's revised solution after peer feedback."""

    changes_made: List[ChangeMade]
    refined_solution: str
    refined_answer: str = Field(description="Concise final answer after refinement.")
    confidence: float = Field(description="Updated confidence in [0, 1].")
