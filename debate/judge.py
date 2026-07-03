"""Stage 4: final judgment.

The judge is a separate persona that did NOT solve the problem. It receives a
single assembled "dossier" (original answers + peer reviews + refined solutions)
rather than the raw objects, so the prompt stays readable and the judge is nudged
to compare like-for-like. Building that dossier is the bulk of the work here.
"""
from typing import Dict, List

from schemas import JudgeDecision, PeerReview, RefinedSolution, SolverSolution
from prompts import judge_system_prompt, judge_user_prompt
from llm_client import structured_call


def _format_dossier(
    labels: List[str],
    originals: Dict[str, SolverSolution],
    reviews_by_target: Dict[str, List[PeerReview]],
    refined: Dict[str, RefinedSolution],
) -> str:
    """Assemble the full case file the judge reads.

    Args:
        labels: ordered list like ["solver_1", "solver_2", "solver_3"].
        originals: {label: SolverSolution}.
        reviews_by_target: {label: [PeerReview, ...]} reviews *about* that label.
        refined: {label: RefinedSolution}.
    """
    parts: List[str] = []
    for label in labels:
        original_solution = originals[label]
        refined_solution = refined[label]

        review_lines: List[str] = []
        for review in reviews_by_target.get(label, []):
            weaknesses = ", ".join(review.weaknesses) or "none"
            review_lines.append(
                f"    - assessment={review.overall_assessment}; weaknesses={weaknesses}"
            )
        reviews_text = "\n".join(review_lines) or "    - (none)"

        parts.append(
            f"=== {label} ===\n"
            f"  Original answer: {original_solution.final_answer}\n"
            f"  Peer reviews of {label}:\n{reviews_text}\n"
            f"  Refined answer: {refined_solution.refined_answer} "
            f"(confidence {refined_solution.confidence})\n"
            f"  Refined solution:\n{refined_solution.refined_solution}"
        )
    return "\n\n".join(parts)


def judge(
    judge_agent: str,
    question: str,
    labels: List[str],
    originals: Dict[str, SolverSolution],
    reviews_by_target: Dict[str, List[PeerReview]],
    refined: Dict[str, RefinedSolution],
) -> JudgeDecision:
    """The judge picks the strongest refined solution."""
    dossier = _format_dossier(labels, originals, reviews_by_target, refined)
    return structured_call(
        judge_system_prompt(judge_agent),
        judge_user_prompt(question, dossier),
        JudgeDecision,
        persona=judge_agent,
    )
