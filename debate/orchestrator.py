from answer_checker import answers_match
from schemas import JudgeDecision
from debate.roles import collect_self_assessments, assign_roles
from debate.solver import solve
from debate.reviewer import review
from debate.refiner import refine
from debate.judge import judge

SOLVER_LABELS = ["solver_1", "solver_2", "solver_3"]


def _fallback_decision(refined):
    labels = list(refined.keys())
    clusters = []
    for label in labels:
        ans = refined[label].refined_answer
        for cluster in clusters:
            if answers_match(ans, refined[cluster[0]].refined_answer):
                cluster.append(label)
                break
        else:
            clusters.append([label])

    clusters.sort(
        key=lambda cl: (len(cl), max(refined[l].confidence for l in cl)),
        reverse=True,
    )
    winner = max(clusters[0], key=lambda l: refined[l].confidence)
    return JudgeDecision(
        winner=winner,
        final_answer=refined[winner].refined_answer,
        confidence=refined[winner].confidence,
        reasoning="Judge call failed; winner chosen by majority of refined answers, "
                  "tie-broken by solver confidence.",
    )


def run_debate(problem, verbose=True):
    question = problem["question"]

    def log(msg):
        if verbose:
            print(msg)

    log("  [Stage 0] collecting role self-assessments...")
    assessments = collect_self_assessments(question)
    solver_agents, judge_agent = assign_roles(assessments)

    label_to_agent = dict(zip(SOLVER_LABELS, solver_agents))
    log(f"  [Stage 0.5] judge={judge_agent}, solvers={solver_agents}")

    log("  [Stage 1] independent solving...")
    originals = {
        label: solve(agent, question) for label, agent in label_to_agent.items()
    }

    log("  [Stage 2] peer review...")
    reviews_by_reviewer = {}          # reviewer_label -> {target_label: PeerReview}
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

    log("  [Stage 3] refinement...")
    refined = {}
    for label in SOLVER_LABELS:
        refined[label] = refine(
            label_to_agent[label], question, originals[label], reviews_by_target[label]
        )

    log("  [Stage 4] judgment...")
    try:
        decision = judge(
            judge_agent, question, SOLVER_LABELS, originals, reviews_by_target, refined
        )
    except Exception as e:                       # judge failed even after retries
        log(f"    judge call failed ({type(e).__name__}); using fallback decision")
        decision = _fallback_decision(refined)

    winner = decision.winner if decision.winner in refined else SOLVER_LABELS[0]
    final_answer = decision.final_answer or refined[winner].refined_answer

    return {
        "id": problem["id"],
        "category": problem["category"],
        "question": question,
        "expected_answer": problem["answer"],
        "acceptable_answers": problem.get("acceptable_answers", []),
        "roles": {"solvers": solver_agents, "judge": judge_agent},
        "label_to_agent": label_to_agent,
        "role_assessments": {a: v.model_dump() for a, v in assessments.items()},
        "initial_solutions": {l: s.model_dump() for l, s in originals.items()},
        "reviews": {
            reviewer: {tgt: rv.model_dump() for tgt, rv in tgts.items()}
            for reviewer, tgts in reviews_by_reviewer.items()
        },
        "refined_solutions": {l: r.model_dump() for l, r in refined.items()},
        "judgment": decision.model_dump(),
        "winner": winner,
        "final_answer": final_answer,
    }
