---
name: planning
description: "Use when scoping or refining a non-trivial change or experiment and you want a repo-local planning workflow that creates or updates one dated canonical plan in `docs/plan/`, reviews it with `main_reviewer`, `big_picture_reviewer`, and `research_reviewer`, and optionally adds compute or devil's-advocate review before implementation starts."
---

# Planning

Use this skill for non-trivial changes, cross-stage refactors, experiment design changes, or any work where the implementation should start from an explicit written plan.

Skip it for tiny edits unless the user explicitly wants planning.

## Components

- the main agent using this skill is the workflow coordinator
- `planner` writes and refines the plan
- `main_reviewer` is the default pragmatic reviewer
- `big_picture_reviewer` is the default system-level reviewer
- `research_reviewer` is the default deep-learning reviewer
- `compute_reviewer` is optional for runtime, memory, dataloading, cluster, or container risk
- `devils_advocate_reviewer` is optional for skeptical challenge when assumptions look fragile

## Canonical Plan Rule

- The plan lives in `docs/plan/`.
- The filename must start with `YYYY-MM-DD_HH-MM-SS_`.
- The plan file is canonical for that initiative.
- Review rounds should refine the same file in place.
- Prefer sending follow-up review packets back to the same `planner` agent thread so the planner keeps context while editing the same file.
- Do not create a new plan file for ordinary review iterations.
- Only fork to a new plan file if the scope changes enough that the original plan is no longer the same initiative.

## Workflow

1. Gather only the relevant code, config, tests, `docs/plan`, and `docs/progress` context.
2. Spawn `planner`.
3. If no plan exists yet, require `planner` to create a dated file in `docs/plan/`.
4. If a plan already exists, require `planner` to refine that exact file in place.
5. Wait for `planner` to finish before review starts.
6. Review the current plan in parallel with:
   - `main_reviewer`
   - `big_picture_reviewer`
   - `research_reviewer`
7. Add `compute_reviewer` only when the plan touches:
   - dataloading
   - runtime throughput
   - memory pressure
   - cluster execution
   - containers
   - large-run operational behavior
8. Add `devils_advocate_reviewer` only when assumptions are brittle, sequencing is risky, or the plan feels deceptively simple.
9. Collect findings grouped by reviewer name. Do not flatten reviewer identity away.
10. Send the grouped findings back to the same `planner` agent thread and require it to edit the same plan file in place.
11. By default, do one serious review-refine round. Run a second only when unresolved important findings remain.
12. Stop when the plan is implementation-ready, not when it is theoretically perfect.

## Rules

- The workflow coordinator owns user communication and stage transitions.
- Reviewers return findings only.
- `planner` is planning-only and must not change non-plan files.
- Keep the plan concise but implementation-ready.
- Use this repo's research style:
  - small increments later during implementation
  - fail-fast over fallback-heavy design
  - defaults in Hydra, not Python
  - remove unnecessary adjacent complexity
- Treat testing and documentation impact as part of planning, not a postscript.
- When the user specifically wants refinement, prefer reusing the same planner agent thread and same plan file path.

## Handoff

When planning is done, carry forward:
- the exact approved plan path in `docs/plan/`
- any accepted assumptions
- any explicitly deferred follow-up that is non-blocking

That approved plan becomes the input to `$only-implementation-testing`.
