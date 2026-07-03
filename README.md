# Multi-LLM Collaborative Debate System

Final project for the Applied LLM Systems course.

**Idea:** three LLM "Solvers" independently solve a hard problem, cross-review each
other's solutions, refine based on feedback, and a fourth LLM "Judge" picks the
best answer. We want to beat a single-LLM baseline and simple majority voting.

Four "agents" = one model driven with four personas (different system prompts +
temperatures), which the assignment allows.

## Planned pipeline

- Stage 0 / 0.5 — role self-assessment + deterministic role assignment
- Stage 1 — independent solving
- Stage 2 — peer review
- Stage 3 — refinement
- Stage 4 — final judgment

## Setup (WIP)

```bash
pip install -r requirements.txt
cp .env.example .env      # add your OpenAI key
```

Structure, evaluation, and run instructions to follow as the stages are built.
