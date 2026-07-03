"""Stage 2: structured peer review."""
from schemas import PeerReview
from prompts import reviewer_system_prompt, reviewer_user_prompt
from llm_client import structured_call


def _format_solution(solution):
    return f"Reasoning:\n{solution.reasoning}\n\nFinal answer: {solution.final_answer}"


def review(reviewer_agent, question, solution_id, solution):
    """`reviewer_agent` critiques the solution labelled `solution_id`."""
    result = structured_call(
        reviewer_system_prompt(reviewer_agent),
        reviewer_user_prompt(question, solution_id, _format_solution(solution)),
        PeerReview,
        persona=reviewer_agent,
    )
    result.solution_id = solution_id  # trust our label over the model's echo
    return result
