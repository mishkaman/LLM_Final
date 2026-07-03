"""Runs the debate (Stages 0-3) for a single problem and returns a record.

Judgment (Stage 4) is not wired in yet: for now the "final" answer is taken from
the highest-confidence refined solution. The Judge replaces that once it lands.
"""
from debate.roles import collect_self_assessments, assign_roles
from debate.solver import solve
from debate.reviewer import review
from debate.refiner import refine

SOLVER_LABELS = ["solver_1", "solver_2", "solver_3"]


def run_debate(problem, verbose=True):
    question = problem["question"]

    def log(msg):
        if verbose:
            print(msg)

    # --- Stage 0 + 0.5: role assignment ------------------------------------
    log("  [Stage 0] collecting role self-assessments...")
    assessments = collect_self_assessments(question)
    solver_agents, judge_agent = assign_roles(assessments)
    label_to_agent = dict(zip(SOLVER_LABELS, solver_agents))
    log(f"  [Stage 0.5] judge={judge_agent}, solvers={solver_agents}")

    # --- Stage 1: independent solutions ------------------------------------
    log("  [Stage 1] independent solving...")
    originals = {
        label: solve(agent, question) for label, agent in label_to_agent.items()
    }

    # --- Stage 2: peer review (each solver reviews the other two) -----------
    log("  [Stage 2] peer review...")
    reviews_by_reviewer = {}
    reviews_by_target = {label: [] for label in SOLVER_LABELS}
    for reviewer_label in SOLVER_LABELS:
        reviewer_agent = label_to_agent[reviewer_label]
        reviews_by_reviewer[reviewer_label] = {}
        for target_label in SOLVER_LABELS:
            if target_label == reviewer_label:
                continue
            rev = review(reviewer_agent, question, target_label, originals[target_label])
            reviews_by_reviewer[reviewer_label][target_label] = rev
            reviews_by_target[target_label].append(rev)

    # --- Stage 3: refinement -----------------------------------------------
    log("  [Stage 3] refinement...")
    refined = {}
    for label in SOLVER_LABELS:
        refined[label] = refine(
            label_to_agent[label], question, originals[label], reviews_by_target[label]
        )

    # No judge yet: provisional winner = highest-confidence refined solution.
    winner = max(SOLVER_LABELS, key=lambda l: refined[l].confidence)
    final_answer = refined[winner].refined_answer

    return {
        "id": problem["id"],
        "category": problem["category"],
        "question": question,
        "expected_answer": problem["answer"],
        "roles": {"solvers": solver_agents, "judge": judge_agent},
        "label_to_agent": label_to_agent,
        "role_assessments": {a: v.model_dump() for a, v in assessments.items()},
        "initial_solutions": {l: s.model_dump() for l, s in originals.items()},
        "reviews": {
            reviewer: {tgt: rv.model_dump() for tgt, rv in tgts.items()}
            for reviewer, tgts in reviews_by_reviewer.items()
        },
        "refined_solutions": {l: r.model_dump() for l, r in refined.items()},
        "winner": winner,
        "final_answer": final_answer,
    }
