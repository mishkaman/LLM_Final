"""Prompt builders for each role/stage.

Kept deliberately separate from orchestration logic so the wording can be
iterated on (and A/B can tune it independently) without touching the pipeline.
Persona flavor is injected once via ``_persona_block`` and reused everywhere, so
a change to how personas are described propagates to all four stages at once.
"""
import config


def _persona_block(agent: str) -> str:
    style = config.PERSONAS[agent]["style"]
    return (
        f"You are Agent {agent}, one of several independent AI reasoners. "
        f"Your working style: {style}."
    )


def role_system_prompt(agent: str) -> str:
    return (
        f"{_persona_block(agent)}\n\n"
        "For a given problem you must decide which role fits you best:\n"
        "- 'solver': produce a full solution to the problem.\n"
        "- 'judge': evaluate other solvers' solutions and pick the best.\n"
        "Assess honestly based on the problem type and your strengths."
    )


def role_user_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "State your preferred_role ('solver' or 'judge'), your confidence in each "
        "role (solver and judge, each a number in [0,1]), and brief reasoning."
    )


def solver_system_prompt(agent: str) -> str:
    return (
        f"{_persona_block(agent)}\n\n"
        "You are a SOLVER. Solve the problem with careful, explicit, step-by-step "
        "reasoning. Verify your work before finalizing. Then give a single concise "
        "final answer with no extra commentary in the final_answer field."
    )


def solver_user_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "Provide: your full step-by-step reasoning, a concise final_answer (just the "
        "answer itself), and your confidence in [0,1]."
    )


def reviewer_system_prompt(agent: str) -> str:
    return (
        f"{_persona_block(agent)}\n\n"
        "You are a peer REVIEWER. Critically evaluate another solver's solution to "
        "the same problem. Be constructive but rigorous: verify each step, look for "
        "arithmetic slips, logical leaps, and unhandled edge cases. Do not rubber-stamp."
    )


def reviewer_user_prompt(question: str, solution_id: str, solution_text: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        f"Solution to review (id = {solution_id}):\n{solution_text}\n\n"
        f"Produce a structured review. Set solution_id to '{solution_id}'. List "
        "strengths and weaknesses; enumerate concrete errors (location, error_type, "
        "description, severity in {{minor, major, critical}}); give suggested_changes; "
        "and set overall_assessment to one of: 'correct', 'promising_but_flawed', "
        "'likely_incorrect'."
    )


def refiner_system_prompt(agent: str) -> str:
    return (
        f"{_persona_block(agent)}\n\n"
        "You are refining YOUR OWN solution after receiving peer reviews. Address "
        "every critique explicitly: fix genuine mistakes, but defend your reasoning "
        "when a critique is wrong. Do not change a correct answer just to appease a reviewer."
    )


def refiner_user_prompt(question: str, own_solution_text: str, reviews_text: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        f"Your original solution:\n{own_solution_text}\n\n"
        f"Peer reviews you received:\n{reviews_text}\n\n"
        "Output: changes_made (for each critique: the critique, your response, and "
        "accepted=true/false), your refined_solution, a concise refined_answer, and "
        "confidence in [0,1]."
    )


def judge_system_prompt(agent: str) -> str:
    # NOTE: the "be concise" instruction here was added after a real run in which
    # the judge rambled, blew past MAX_OUTPUT_TOKENS, and truncated its own JSON.
    return (
        f"{_persona_block(agent)}\n\n"
        "You are the impartial JUDGE. You did not solve the problem yourself. Read all "
        "three refined solutions (with the original solutions and peer reviews for "
        "context), reason about which is most likely correct and best-justified, and "
        "select exactly one winner.\n\n"
        "Keep your reasoning concise (at most 3-4 sentences). Do NOT restate or copy the "
        "solutions back; just explain your choice. Put only the short final answer in "
        "final_answer."
    )


def judge_user_prompt(question: str, dossier_text: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        f"{dossier_text}\n\n"
        "Select the winner: one of 'solver_1', 'solver_2', 'solver_3'. Copy the winning "
        "solution's answer into final_answer. Provide confidence in [0,1] and reasoning."
    )
