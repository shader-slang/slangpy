# Codex Execution Plans (ExecPlans)

This document describes the requirements for an execution plan ("ExecPlan"), a design document that a coding agent can follow to deliver a working feature or system change. Treat the reader as a complete beginner to this repository: they have only the current working tree and the single ExecPlan file you provide. There is no memory of prior plans and no external context.

## How to use ExecPlans and PLANS.md

When authoring an executable specification (ExecPlan), follow PLANS.md to the letter. If it is not in your context, refresh your memory by reading the entire PLANS.md file. Be thorough in reading and re-reading source material to produce an accurate specification. When creating a spec, start from the skeleton and flesh it out as you do your research.

When implementing an executable specification, do not prompt the user for "next steps"; proceed to the next milestone. Keep all sections up to date, add or split entries in the list at every stopping point to affirmatively state progress made and next steps, and resolve ambiguities autonomously where the repository context supports doing so. Record progress frequently in the plan; create git commits only when the active workflow or user request calls for commits.

When discussing an executable specification, record decisions in a log in the spec for posterity. It should be unambiguously clear why any change to the specification was made. ExecPlans are living documents, and it should always be possible to restart from only the ExecPlan and no other work.

When researching a design with challenging requirements or significant unknowns, use milestones to implement proof of concepts or toy implementations that validate whether the proposal is feasible. Read the source code of relevant libraries by finding or acquiring them, research deeply, and include prototypes to guide the full implementation.

## Requirements

Non-negotiable requirements:

* Every ExecPlan must be fully self-contained. Self-contained means that in its current form it contains all knowledge and instructions needed for a novice to succeed.
* Every ExecPlan is a living document. Contributors are required to revise it as progress is made, as discoveries occur, and as design decisions are finalized. Each revision must remain fully self-contained.
* Every ExecPlan must enable a complete novice to implement the feature end-to-end without prior knowledge of this repository.
* Every ExecPlan must produce a demonstrably working behavior, not merely code changes to meet a definition.
* Every ExecPlan must define every term of art in plain language or avoid using it.

Purpose and intent come first. Begin by explaining, in a few sentences, why the work matters from a user's perspective: what someone can do after this change that they could not do before, and how to see it working. Then guide the reader through the exact steps to achieve that outcome, including what to edit, what to run, and what they should observe.

The agent executing your plan can list files, read files, search, run the project, and run tests. It does not know any prior context and cannot infer what you meant from earlier milestones. Repeat every assumption you rely on. Do not point to external blogs or docs; if knowledge is required, embed it in the plan itself in your own words. If an ExecPlan builds upon a prior ExecPlan and that file is checked in, incorporate it by reference. If it is not checked in, include all relevant context from that plan.

## Formatting

Format and envelope are simple and strict. Each ExecPlan must be one single fenced code block labeled as `md` that begins and ends with triple backticks. Do not nest additional triple-backtick code fences inside; when you need to show commands, transcripts, diffs, or code, present them as indented blocks within that single fence. Use indentation for clarity rather than code fences inside an ExecPlan. Use two newlines after every heading, use correct Markdown heading syntax, and keep the prose plain.

When writing an ExecPlan to a Markdown file where the content of the file is only the single ExecPlan, omit the outer triple backticks.

Prefer sentences over lists. Avoid checklists, tables, and long enumerations unless brevity would obscure meaning. Checklists are permitted only in the Progress section, where they are mandatory. Narrative sections must remain prose-first.

## Guidelines

Self-containment and plain language are paramount. If you introduce a phrase that is not ordinary English, define it immediately and explain how it appears in this repository. Do not say "as defined previously" or "according to the architecture doc." Include the needed explanation in the plan, even if it repeats repository guidance.

Avoid common failure modes. Do not rely on undefined jargon. Do not describe a feature so narrowly that the resulting code compiles but does nothing meaningful. Do not outsource key decisions to the reader. When ambiguity exists, resolve it in the plan itself and explain why you chose that path. Err on the side of over-explaining user-visible effects and under-specifying incidental implementation details.

Anchor the plan with observable outcomes. State what the user can do after implementation, the commands to run, and the outputs they should see. Acceptance should be phrased as behavior a human can verify rather than internal attributes. If a change is internal, explain how its impact can still be demonstrated, for example by running tests that fail before and pass after, or by showing a scenario that uses the new behavior.

Specify SlangPy repository context explicitly. Name files with full repository-relative paths, name functions and modules precisely, and describe where new files should be created. If touching multiple areas, include a short orientation paragraph that explains how those parts fit together. For example, changes to functional API behavior may involve `slangpy/core/calldata.py`, `slangpy/bindings/*`, `src/slangpy_ext/utils/slangpyfunction.cpp`, `src/slangpy_ext/utils/slangpy.cpp`, and tests under `slangpy/tests/`.

When running commands, show the working directory and exact command line. SlangPy generally requires a build before tests. Typical commands include:

    cd C:\sw\slangpy
    cmake --build --preset windows-msvc-debug
    pytest slangpy/tests -v
    pytest samples/tests -vra
    python tools/ci.py unit-test-cpp
    pre-commit run --all-files

Be idempotent and safe. Write the steps so they can be run multiple times without causing damage or drift. If a step can fail halfway, include how to retry or adapt. If a migration or destructive operation is necessary, spell out backups or safe fallbacks. Prefer additive, testable changes that can be validated as you go.

Validation is not optional. Include instructions to build, run tests, start the system if applicable, and observe it doing something useful. Describe comprehensive testing for any new features or capabilities. Include expected outputs and error messages so a novice can tell success from failure. State the exact test commands appropriate to SlangPy and how to interpret their results.

Capture evidence. When your steps produce terminal output, short diffs, or logs, include concise examples. Keep them focused on what proves success. If you need to include a patch, prefer file-scoped diffs or small excerpts that a reader can recreate by following the instructions rather than pasting large blobs.

## Milestones

Milestones are narrative, not bureaucracy. If you break the work into milestones, introduce each with a brief paragraph that describes the scope, what will exist at the end of the milestone that did not exist before, the commands to run, and the acceptance you expect to observe.

Each milestone must be independently verifiable and must incrementally implement the overall goal of the execution plan. Progress and milestones are distinct: milestones tell the story, progress tracks granular work. Both must exist.

## Living plans and design decisions

ExecPlans are living documents. As you make key design decisions, update the plan to record both the decision and the thinking behind it. Record all decisions in the Decision Log section.

ExecPlans must contain and maintain a Progress section, a Surprises and Discoveries section, a Decision Log, and an Outcomes and Retrospective section. These are not optional.

When you discover compiler behavior, platform constraints, GPU backend differences, performance tradeoffs, unexpected bugs, or inverse/unapply semantics that shape your approach, capture those observations in Surprises and Discoveries with concise evidence. Test output is ideal.

If you change course mid-implementation, document why in the Decision Log and reflect the implications in Progress. Plans are guides for the next contributor as much as checklists for you.

At completion of a major task or the full plan, write an Outcomes and Retrospective entry summarizing what was achieved, what remains, and lessons learned.

## Prototyping milestones and parallel implementations

It is acceptable, and often encouraged, to include explicit prototyping milestones when they de-risk a larger change. Examples include adding a low-level operator to validate feasibility, trying a new marshalling path for one type before generalizing it, or testing generated Slang code in a narrow scenario before wiring it into the full functional API. Keep prototypes additive and testable. Clearly label the scope as prototyping, describe how to run and observe results, and state the criteria for promoting or discarding the prototype.

Prefer additive code changes followed by subtractions that keep tests passing. Parallel implementations are fine when they reduce risk or enable tests to continue passing during a large migration. Describe how to validate both paths and how to retire one safely with tests.

## Skeleton of a Good ExecPlan

    # <Short, action-oriented description>

    This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

    This plan follows `.agents/PLANS.md` from the repository root.

    ## Purpose / Big Picture

    Explain in a few sentences what someone gains after this change and how they can see it working. State the user-visible behavior you will enable.

    ## Progress

    Use a list with checkboxes to summarize granular steps. Every stopping point must be documented here, even if it requires splitting a partially completed task into done and remaining work. This section must always reflect the actual current state of the work.

    - [x] (2026-05-29 08:00Z) Example completed step.
    - [ ] Example incomplete step.
    - [ ] Example partially completed step (completed: X; remaining: Y).

    Use timestamps to measure rates of progress.

    ## Surprises and Discoveries

    Document unexpected behaviors, bugs, optimizations, or insights discovered during implementation. Provide concise evidence.

    - Observation: ...
      Evidence: ...

    ## Decision Log

    Record every decision made while working on the plan in this format:

    - Decision: ...
      Rationale: ...
      Date/Author: ...

    ## Outcomes and Retrospective

    Summarize outcomes, gaps, and lessons learned at major milestones or at completion. Compare the result against the original purpose.

    ## Context and Orientation

    Describe the current state relevant to this task as if the reader knows nothing. Name the key files and modules by full path. Define any non-obvious term you will use. Do not refer to prior plans unless they are checked into this repository and summarized here.

    ## Plan of Work

    Describe, in prose, the sequence of edits and additions. For each edit, name the file and location, such as function or module, and what to insert or change. Keep it concrete and minimal.

    ## Concrete Steps

    State the exact commands to run and where to run them. When a command generates output, show a short expected transcript so the reader can compare.

    ## Validation and Acceptance

    Describe how to exercise the system and what to observe. Phrase acceptance as behavior, with specific inputs and outputs. If tests are involved, say which project test command to run and what should pass; identify any new test that fails before the change and passes after.

    ## Idempotence and Recovery

    If steps can be repeated safely, say so. If a step is risky, provide a safe retry or rollback path. Keep the environment clean after completion.

    ## Artifacts and Notes

    Include the most important transcripts, diffs, or snippets as indented examples. Keep them concise and focused on what proves success.

    ## Interfaces and Dependencies

    Be prescriptive. Name the libraries, modules, and services to use and why. Specify the types, interfaces, and function signatures that must exist at the end of the milestone. Prefer stable names and paths such as `slangpy.core.calldata.CallData` or `src/slangpy_ext/utils/slangpy.cpp`.

When you revise a plan, ensure your changes are reflected across all sections, including the living document sections. Add a short note near the bottom of the plan describing what changed and why. ExecPlans must describe not just what to do, but why the work is shaped that way.
