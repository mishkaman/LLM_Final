"""Compute system-level and comparative metrics from debate records."""
from collections import defaultdict

from answer_checker import is_correct
from evaluation.baselines import (
    single_llm_answer,
    majority_vote_answer,
    all_solvers_agree,
)


def _correct(answer, record):
    if answer is None:
        return False
    return is_correct(answer, record["expected_answer"], record.get("acceptable_answers"))


def compute_metrics(records):
    """Return a dict of all metrics required by the assignment.

    Records with an "error" key (a failed run) are skipped from accuracy
    numerators/denominators but counted in `n_failed`.
    """
    valid = [r for r in records if "error" not in r]
    n = len(valid)
    n_failed = len(records) - n

    if n == 0:
        return {"n_problems": 0, "n_failed": n_failed,
                "note": "No successful records to evaluate."}

    system_correct = 0
    single_correct = 0
    majority_correct = 0
    consensus_count = 0
    improvement_count = 0          # winner's initial wrong -> refined right
    disagree_total = 0             # problems where solvers did NOT all agree
    disagree_judge_correct = 0     # ...and the judge's final answer was correct

    per_category = defaultdict(lambda: {"total": 0, "system_correct": 0})

    for r in valid:
        sys_ok = _correct(r["final_answer"], r)
        single_ok = _correct(single_llm_answer(r), r)
        majority_ok = _correct(majority_vote_answer(r), r)

        system_correct += sys_ok
        single_correct += single_ok
        majority_correct += majority_ok

        cat = per_category[r["category"]]
        cat["total"] += 1
        cat["system_correct"] += sys_ok

        agree = all_solvers_agree(r)
        consensus_count += agree

        # Improvement: the winning solver was wrong initially but right after refinement.
        winner = r.get("winner")
        init = r.get("initial_solutions", {}).get(winner, {})
        winner_initial = init.get("final_answer")
        if winner_initial is not None:
            if (not _correct(winner_initial, r)) and sys_ok:
                improvement_count += 1

        # Judge accuracy when there is genuine disagreement to arbitrate.
        if not agree:
            disagree_total += 1
            disagree_judge_correct += sys_ok

    def pct(x):
        return round(100.0 * x / n, 1)

    metrics = {
        "n_problems": n,
        "n_failed": n_failed,
        "model": None,  # filled in by the runner
        "accuracy": {
            "single_llm_baseline": pct(single_correct),
            "majority_vote_baseline": pct(majority_correct),
            "debate_system": pct(system_correct),
        },
        "consensus_rate": pct(consensus_count),
        "improvement_rate": pct(improvement_count),
        "judge_accuracy_on_disagreement": (
            round(100.0 * disagree_judge_correct / disagree_total, 1)
            if disagree_total else None
        ),
        "n_disagreements": disagree_total,
        "per_category_accuracy": {
            cat: round(100.0 * v["system_correct"] / v["total"], 1)
            for cat, v in per_category.items()
        },
    }
    return metrics
