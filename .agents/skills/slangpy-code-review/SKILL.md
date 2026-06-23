---
name: slangpy-code-review
description: Multi-agent SlangPy code review using predefined specialist sub-agents. Use when the user explicitly asks for sub-agents, parallel review, specialist reviewers, or a deeper delegated code review of local diffs, branches, PRs, MRs, or specific SlangPy files.
---

# SlangPy Code Review

Use this skill to run a coordinated multi-agent review. The parent agent is the review captain: it defines the review scope, dispatches specialist reviewers, deduplicates findings, and presents a concise final review.

## Predefined Reviewers

Use these repo-local agent definitions:

- `.agents/agents/slangpy-correctness-reviewer.md`
- `.agents/agents/slangpy-functional-api-reviewer.md`
- `.agents/agents/slangpy-api-bindings-reviewer.md`
- `.agents/agents/slangpy-test-reviewer.md`
- `.agents/agents/slangpy-performance-reviewer.md`
- `.agents/agents/slangpy-maintainability-reviewer.md`
- `.agents/agents/slangpy-build-ci-reviewer.md`

If the Codex runtime exposes these names as sub-agent types, spawn those exact types. If it only exposes generic sub-agent types, spawn generic read-only agents and include the matching agent definition path or contents in each prompt.

## Workflow

1. Establish review scope, defaulting to a comparison against `origin/main` when the user does not specify one. Common scopes include:
   - local working tree diff,
   - branch comparison to `origin/main`,
   - PR/MR diff,
   - explicit files,
   - or an explicit patch supplied by the user.

2. Inspect enough local context to create a useful reviewer packet:
   - changed file list,
   - relevant diff,
   - nearby code patterns,
   - tests touched or missing,
   - build files affected.

3. Run all predefined reviewers in parallel, giving them the same scope and context. Each reviewer should return findings in the specified format.

4. Collate results:
   - Deduplicate overlapping findings.
   - Keep only concrete, defensible issues.
   - Drop vague style preferences unless they indicate real risk.
   - Resolve conflicting reviewer opinions by checking the code locally.
   - Sort by severity.

5. Report in code-review style:
   - Findings first.
   - Then open questions.
   - Then test gaps and verification notes.
   - Summary last and brief.

## Reviewer Prompt Template

```text
Use the predefined reviewer in <agent-definition-path>.

Review target:
<scope and diff/files>

Context:
<project-specific notes, relevant conventions, nearby files, or test/build notes>

Rules:
- Read only. Do not edit files.
- Return only actionable findings.
- Include file and line/function where possible.
- Explain impact and suggested fix.
- Say "No findings" if no actionable issue is found.
```

## Final Output

Use this shape:

```markdown
**Findings**
- [severity] file:line - Issue, impact, and suggested fix.

**Open Questions**
- Question or assumption, if any.

**Test Gaps**
- Missing or recommended tests.

**Verification**
- Commands run, commands not run, and why.
```

If there are no findings, say that clearly and still mention any residual test or verification gaps.
