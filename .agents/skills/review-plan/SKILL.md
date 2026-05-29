---
name: review-plan
description: Review a plan using sub agents and provide feedback on how to improve it.
---

# Review Plan

Use this skill to review an implementation plan, ExecPlan, design note, or proposed task breakdown before coding. Prefer this for plans that touch multiple SlangPy layers, GPU behavior, the functional API, build configuration, or user-visible public APIs.

Run 5 sub-agents to review the plan and provide feedback. The agents have these roles:

- correctness reviewer: review the plan for correctness and identify errors, contradictions, or invalid assumptions.
- completeness reviewer: review the plan for missing steps, missing context, or missing validation.
- clarity reviewer: review the plan for confusing wording, undefined terms, or instructions a new contributor could misread.
- efficiency reviewer: review the plan for unnecessary steps, avoidable churn, or inefficient sequencing.
- performance reviewer: review the plan for potential CPU or GPU performance issues.

Once the agents have reviewed the plan, collate the feedback into a concise set of actionable improvements. Deduplicate overlapping advice, ground suggestions in the plan text, and distinguish required fixes from optional refinements.

Final response shape:

- Key issues that should be fixed before implementation.
- Suggested edits or additions.
- Residual risks and validation gaps.
- A direct question asking whether the user wants the feedback applied.
