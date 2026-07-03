"""Stage 1: independent solution generation."""
from schemas import SolverSolution
from prompts import solver_system_prompt, solver_user_prompt
from llm_client import structured_call


def solve(agent, question):
    """One solver produces a complete, independent solution."""
    return structured_call(
        solver_system_prompt(agent),
        solver_user_prompt(question),
        SolverSolution,
        persona=agent,
    )
