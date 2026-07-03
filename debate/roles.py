"""Stage 0 (self-assessment) and Stage 0.5 (deterministic role assignment)."""
import config
from schemas import RoleSelfAssessment
from prompts import role_system_prompt, role_user_prompt
from llm_client import structured_call


def collect_self_assessments(question):
    """Stage 0: every agent rates its own fit for Solver vs Judge."""
    return {
        agent: structured_call(
            role_system_prompt(agent),
            role_user_prompt(question),
            RoleSelfAssessment,
            persona=agent,
        )
        for agent in config.AGENT_ORDER
    }


def assign_roles(assessments):
    """Stage 0.5: deterministic -> 1 Judge + 3 Solvers.

    Judge = the agent with the largest (judge - solver) confidence margin, i.e.
    the one that most prefers judging. Ties break by higher judge confidence,
    then by AGENT_ORDER. Fully deterministic given the same assessments.
    """
    def judge_key(agent):
        c = assessments[agent].confidence
        # negative index so earlier agents win ties under max()
        return (c.judge - c.solver, c.judge, -config.AGENT_ORDER.index(agent))

    judge = max(config.AGENT_ORDER, key=judge_key)
    solvers = [a for a in config.AGENT_ORDER if a != judge]
    return solvers, judge
