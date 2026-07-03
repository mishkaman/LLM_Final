"""Entry point: run the debate system over the problem dataset.

Examples:
    python run_debate.py --limit 3      # cheap smoke test (first 3 problems)
    python run_debate.py                # full 25-problem run
"""
import argparse
import json

import config
from debate.orchestrator import run_debate


def load_problems():
    with open(config.PROBLEMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["problems"]


def main():
    parser = argparse.ArgumentParser(description="Run the multi-LLM debate system.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N problems (for a cheap test).")
    parser.add_argument("--out", type=str, default=config.DEBATE_RESULTS_PATH,
                        help="Where to write the results JSON.")
    args = parser.parse_args()

    problems = load_problems()
    if args.limit is not None:
        problems = problems[: args.limit]

    print(f"Running debate on {len(problems)} problem(s) with model "
          f"'{config.MODEL_NAME}'...\n")

    results = []
    for i, problem in enumerate(problems, start=1):
        print(f"[{i}/{len(problems)}] {problem['id']} ({problem['category']})")
        try:
            record = run_debate(problem)
            final = record["final_answer"]
            print(f"  -> final answer: {final!r} (expected {problem['answer']!r})\n")
            results.append(record)
        except Exception as e:                       # keep going on a single failure
            print(f"  !!! failed: {e}\n")
            results.append({
                "id": problem["id"],
                "category": problem["category"],
                "question": problem["question"],
                "expected_answer": problem["answer"],
                "acceptable_answers": problem.get("acceptable_answers", []),
                "error": str(e),
            })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} records to {args.out}")


if __name__ == "__main__":
    main()
