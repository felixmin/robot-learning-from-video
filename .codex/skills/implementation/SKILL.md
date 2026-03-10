---
name: implementation
description: "Use when an approved dated plan already exists in `docs/plan/` and you want the repo-local workflow for implementation, targeted testing, optional research or devil's-advocate checks, final review, and progress-note updates without re-entering planning unless the scope changes materially."
---

# Implementation

Use this skill when there is already an approved dated plan in `docs/plan/` and the job is to execute it, validate it, and close it out cleanly.

If there is no approved plan and the task is non-trivial, use `$planning` first.

## Components

- the main agent using this skill is the workflow coordinator
- `implementer` owns code, config, and doc changes
- `test_runner` owns independent validation and exact command reporting
- `final_reviewer` owns the final plan-versus-implementation review
- `research_reviewer` is optional for ML-heavy logic, training behavior, metrics, sampling, or evaluation risk
- `devils_advocate_reviewer` is optional for risky refactors, brittle sequencing, or suspiciously tidy implementations

## Workflow

1. Load the approved plan path from `docs/plan/`.
2. Gather only the code, config, tests, and docs needed for that plan.
3. Spawn `implementer` with the approved plan path and the relevant write scope.
4. Wait for `implementer` to finish its current implementation pass.
5. Spawn `test_runner` to validate the implemented scope with targeted checks.
6. If `test_runner` finds failures, send the findings back to the same `implementer` agent thread and continue the implement-test loop.
7. Add `research_reviewer` after testing when the change materially affects:
   - objectives or losses
   - data sampling or split logic
   - metrics or evaluation claims
   - train-time versus inference-time behavior
8. Add `devils_advocate_reviewer` when the change is risky enough that one more skeptical pass is worth the cost.
9. Once implementation and validation are in good shape, spawn `final_reviewer`.
10. If final review finds issues, send them back to the same `implementer` agent thread and repeat the loop.
11. Finish by reporting:
   - the approved plan path
   - what changed
   - what was tested
   - what docs or progress notes were updated
   - any remaining risks or explicitly deferred non-blocking work

## Rules

- This skill does not create or redesign the plan. It executes the approved one.
- If a real repo fact invalidates the approved plan, stop pretending the plan still fits and switch back to `$planning`.
- Reuse the same `implementer` agent thread across implementation iterations when possible so the agent keeps local context.
- `test_runner` validates and reports; it does not edit source files.
- `final_reviewer` returns findings and verdict only.
- Prefer targeted checks before broad suites.
- Keep the repo's implement-run-analyze loop tight and concrete.
- Use `docs/progress/` for durable execution evidence when the task is substantial enough to warrant it.

## Completion Standard

The workflow is complete when:
- the approved plan is implemented or any justified deviation is explicitly recorded
- relevant tests or smoke checks were run
- stale code was removed where appropriate
- docs are not left misleading
- any durable progress note that the change merits has been written
