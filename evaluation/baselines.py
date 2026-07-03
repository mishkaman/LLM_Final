"""Baselines derived from the stored debate records (no extra API calls).

- Single-LLM baseline: what one model says on its own = solver_1's *initial*
  (pre-refinement) answer. That is exactly "just ask the model once".
- Majority-vote baseline: the most common answer among the three solvers'
  initial solutions, using the answer_checker to cluster equivalent answers.

Deriving both from the same run keeps the comparison fair and free -- no separate
baseline pass, no extra tokens, same problems.
"""
from answer_checker import answers_match

SOLVER_LABELS = ["solver_1", "solver_2", "solver_3"]


def _initial_answers(record):
    return [
        record["initial_solutions"][label]["final_answer"]
        for label in SOLVER_LABELS
        if label in record.get("initial_solutions", {})
    ]


def single_llm_answer(record):
    """One unaided model call = the first solver's initial answer."""
    sols = record.get("initial_solutions", {})
    if "solver_1" in sols:
        return sols["solver_1"]["final_answer"]
    return None


def majority_vote_answer(record):
    """Most common initial answer; ties break toward the first cluster seen."""
    answers = _initial_answers(record)
    if not answers:
        return None

    clusters = []  # each: list of equivalent answer strings
    for a in answers:
        for cluster in clusters:
            if answers_match(a, cluster[0]):
                cluster.append(a)
                break
        else:
            clusters.append([a])

    clusters.sort(key=len, reverse=True)
    return clusters[0][0]


def all_solvers_agree(record):
    """True if all three initial answers are mutually equivalent (consensus)."""
    answers = _initial_answers(record)
    if len(answers) < 2:
        return False
    first = answers[0]
    return all(answers_match(first, a) for a in answers[1:])
