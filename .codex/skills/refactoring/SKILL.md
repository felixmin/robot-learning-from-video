---
name: refactoring
description: "Use when handling a non-trivial refactor where behavior should stay stable while structure improves. This planning-only workflow diagnoses the real problem, freezes must-preserve behavior, explores target designs, checks current Hydra defaults and config docs, and writes or refines one canonical plan in `docs/plan/` through narrowly-scoped agents."
---

# Refactoring

Use this skill for non-trivial structural refactors, boundary cleanups, responsibility splits, dependency untangling, config-surface reductions, and similar work where behavior should largely stay the same while the code shape improves.

This skill stops at an approved plan. It does not implement the refactor.

Skip it for tiny local cleanups unless the user explicitly wants the full refactor planning workflow.

## Reused Roles

Reuse these existing roles because they already have the right boundaries:

- `planner` writes or refines the one canonical plan file in `docs/plan/`
- `main_reviewer` gives the default pragmatic review of the refactor plan and its implementation shape
- `big_picture_reviewer` is optional for cross-package, cross-stage, or config-surface consequences
- `research_reviewer` is optional when the refactor touches ML behavior, metrics, data flow, or evaluation semantics
- `compute_reviewer` is optional when the refactor affects runtime, memory, dataloading, caching, or cluster execution
- `devils_advocate_reviewer` critically questions whether the planned implementation is too clever, too abstract, or riskier than necessary

## New Specialized Roles

These roles exist because the early refactor steps need stricter separation than the generic planning workflow provides:

- `refactor_analyst` diagnoses the current structural problem and root cause
- `behavior_guardian` freezes the must-preserve behavior, contracts, and regression surface
- `refactor_architect` explores candidate structures and recommends the migration shape

## Canonical Artifact Rule

- The canonical artifact is still one dated markdown plan file in `docs/plan/`.
- The filename must start with `YYYY-MM-DD_HH-MM-SS_`.
- Use one plan file per initiative and refine that same file in place.
- The three specialized early-step agents do not write the plan file. They return inputs that feed the canonical plan through `planner`.

## Workflow

1. Gather only the relevant code, config, tests, and docs for the suspected refactor scope.
2. Read the current Hydra defaults and config documentation before planning:
   - `config/config.yaml`
   - the relevant `config/experiment/_*.yaml` or task-specific config bases
   - any nearby docs that explain the current Hydra composition or override surface
3. Spawn `refactor_analyst` and `behavior_guardian` in parallel on the same scope.
4. Wait for both before design begins. Do not let architecture work start from a vague diagnosis or an unfrozen contract.
5. Spawn `refactor_architect` with:
   - the relevant repo context
   - the grouped output from `refactor_analyst`
   - the grouped output from `behavior_guardian`
   - the current Hydra defaults/docs context
6. Require `refactor_architect` to fan out to multiple candidate shapes before choosing one. Do not accept a single unexplored design when the structure is non-obvious.
7. Require `refactor_architect` to explicitly question the implementation shape:
   - is it the smallest understandable change that solves the real problem?
   - does it reduce or at least avoid increasing lines of code?
   - does it avoid adding abstractions whose only benefit is conceptual neatness?
   - does it remove legacy code immediately instead of preserving parallel old paths?
   - does it avoid fallbacks and compatibility shims unless they are truly unavoidable?
   - will a future reader understand the final structure quickly?
8. Spawn `planner` with the grouped outputs from:
   - `refactor_analyst`
   - `behavior_guardian`
   - `refactor_architect`
9. If no plan exists yet, require `planner` to create a dated file in `docs/plan/`. If one already exists, require it to refine that exact file in place.
10. Require the plan to call out:
   - the real structural problem
   - the frozen behavior surface
   - the preferred target structure and rejected alternatives
   - the expected Hydra defaults/doc impact
   - why the implementation should stay simple and comprehensible
   - how the change aims to reduce or at least not grow total code size
   - which legacy code paths should be removed immediately
   - which defaults must live in Hydra rather than Python code
11. Review the refactor plan in parallel with:
   - `main_reviewer`
   - `devils_advocate_reviewer`
   - `big_picture_reviewer` when the refactor crosses package, stage, or config boundaries
   - `research_reviewer` when semantics around data, metrics, objectives, or evaluation may drift
   - `compute_reviewer` when the refactor may change performance, memory, dataloading, caching, or cluster behavior
12. In review, explicitly challenge:
   - whether the planned implementation is smaller and clearer than the status quo
   - whether any new abstraction is truly necessary
   - whether the plan matches the current Hydra defaults and surrounding config docs
   - whether the plan removes outdated code immediately instead of carrying it forward
   - whether the plan relies on fallbacks, compatibility layers, or dual paths that should simply not exist
   - whether the migration path is understandable instead of architecturally ornate
13. Send grouped review findings back to the same `planner` thread and require it to edit the same plan file in place.
14. By default, do one serious review-refine round. Do a second only when important findings remain unresolved.
15. Finish by reporting the approved plan path, accepted assumptions, Hydra/config doc implications, and any residual risks or deferred follow-ups.

## Stage Boundaries

- `refactor_analyst` owns diagnosis only.
- `behavior_guardian` owns must-preserve behavior only.
- `refactor_architect` owns target structure and migration shape only.
- `planner` owns the canonical `docs/plan/` artifact only.
- Reviewers return findings only.

Do not blur these roles. If one stage discovers a problem that belongs to an earlier stage, route it back explicitly instead of letting the current agent silently take over another job.

## Rules

- Do not skip the behavior-freezing step just because the code looks obviously wrong.
- Do not let `refactor_analyst` prescribe the architecture.
- Do not let `behavior_guardian` redesign ownership or module boundaries.
- Do not let `refactor_architect` redefine the frozen behavior surface to make the design cleaner.
- Do not implement from this skill. Stop at the approved plan.
- Prefer structural simplification over abstraction growth.
- Prefer smaller code over larger code unless a larger change is clearly justified.
- Prefer deleting or collapsing code paths over introducing new layers.
- Prefer removing legacy code immediately over preserving backward-compatible internal paths.
- Do not plan fallback-heavy migrations unless there is a concrete external compatibility requirement.
- Do not keep outdated code around "just in case" when the plan can replace it cleanly.
- Prefer direct, easily comprehensible structures over clever abstractions.
- Keep defaults in Hydra rather than Python.
- Avoid Python-side defaults for shared behavior when Hydra should be the source of truth.
- Check the current Hydra defaults lists and config comments before proposing config movement or schema changes.
- Call out any plan that would make the Hydra surface harder to understand.
- Match the repo style: fail fast, keep changes minimal, and remove unnecessary adjacent complexity.

## Fan-Out Guidance

Use fan-out where it creates leverage and avoid it where it creates noise:

- `refactor_analyst`: fan out root-cause hypotheses when the pain could come from several different structural issues
- `behavior_guardian`: enumerate externally important behaviors and regression-sensitive edges, but do not brainstorm architecture here
- `refactor_architect`: fan out 2-4 candidate shapes and compare them on clarity, migration risk, dependency direction, repo fit, future change cost, code-size impact, and abstraction cost
- review: challenge whether the chosen design is simpler than the obvious alternative, not merely more elegant on paper
- review: challenge any fallback, compatibility layer, or retained legacy path as guilty until proven necessary

## Completion Standard

The workflow is complete when:

- the actual problem was diagnosed rather than guessed
- the must-preserve behavior was made explicit
- a target structure was chosen from real alternatives
- the plan explicitly checks the current Hydra defaults and config docs
- the plan explains why the implementation should remain simple and comprehensible
- the plan biases toward reducing code and avoiding over-abstraction
- the plan removes outdated code promptly instead of preserving legacy paths without a hard reason
- the plan keeps defaults in Hydra instead of drifting into Python code
- one canonical plan file in `docs/plan/` records the refactor
- review has challenged whether the proposed implementation is too large, too abstract, or too clever
- review has challenged whether any fallback or compatibility layer should exist at all
- docs are not left misleading
