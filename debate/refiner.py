"""Stage 3: refinement based on peer feedback."""
from schemas import RefinedSolution
from prompts import refiner_system_prompt, refiner_user_prompt
from llm_client import structured_call


def _format_own_solution(solution):
    return f"Reasoning:\n{solution.reasoning}\n\nFinal answer: {solution.final_answer}"


def _format_reviews(reviews):
    blocks = []
    for i, r in enumerate(reviews, start=1):
        errors = "; ".join(
            f"[{e.severity}] {e.location}: {e.description}" for e in r.errors
        ) or "none listed"
        blocks.append(
            f"Review {i} (assessment: {r.overall_assessment})\n"
            f"  Strengths: {', '.join(r.strengths) or 'none'}\n"
            f"  Weaknesses: {', '.join(r.weaknesses) or 'none'}\n"
            f"  Errors: {errors}\n"
            f"  Suggested changes: {', '.join(r.suggested_changes) or 'none'}"
        )
    return "\n\n".join(blocks)


def refine(agent, question, own_solution, reviews):
    """A solver revises its own solution given the two reviews it received."""
    return structured_call(
        refiner_system_prompt(agent),
        refiner_user_prompt(
            question,
            _format_own_solution(own_solution),
            _format_reviews(reviews),
        ),
        RefinedSolution,
        persona=agent,
    )
