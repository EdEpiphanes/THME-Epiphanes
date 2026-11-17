---
name: lean-orchestrate
description: Token-lean model routing and orchestration for multi-step work. Use whenever a task has several steps, mixes research with coding or UI, needs a plan or a review, or the user mentions orchestrating, delegating, subagents, an advisor, saving tokens, speed, or the weekly limit. Routes haiku to reading and research, sonnet to implementation, and opus only as a short advisor at decision points. Use it even when the user does not name it, to keep spend low.
---

# Lean orchestration

Goal: the correct result with the fewest tokens. Spend leaks through four defects: re-reading, over-long output, rework, and the wrong model for the job. Count these; remove them.

## Routing: the cheapest model that clears the bar

| Work | Model | How |
|---|---|---|
| Search or read many files, web research, summarise logs/docs, classify, extract, draft prose | `haiku` | `Agent` with `model: "haiku"` |
| Implement, edit, debug, test, build UI | `sonnet` | The main session does it inline |
| Plan a risky change, review a finished diff, decide between trade-offs, unstick after 2 failed tries | `opus` | `Agent` with `model: "opus"`, as an advisor (below) |

Use model aliases (`haiku`, `sonnet`, `opus`), never version strings, so routing tracks the newest release.

## When to delegate, and when not to

A subagent starts cold and re-derives context, so delegating small work costs more than doing it.

- Delegate when the output is noisy (big greps, logs, many pages) and only a digest should enter your context, or when pieces are independent (spawn them together in one turn).
- Do it inline when it takes under about 3 tool calls or you already hold the context.

## Delegation brief (every spawn)

Give: the goal, exact paths or URLs, what to return, and a size cap ("max 120 words, bullets, `file:line` refs, no code dumps"). Add "report, do not fix" for read-only jobs. A tight brief is the cheapest lever, because a vague one causes rework.

## Advisor protocol (opus)

Consult at most 3 times per task: once before (plan), once mid-way only if stuck, once after (review). Send the goal, constraints, the 2-3 options you considered, and one question. Ask for a verdict of at most 200 words. Skip the advisor for reversible, well-understood changes, since its value is in decisions you cannot cheaply undo.

## Kaizen loop (lightweight Six Sigma)

1. **Define**: state "done" in one verifiable line before starting (a test, a command, a screenshot).
2. **Measure**: for any task that used delegation or had rework, append one line to `.claude/lean-ledger.md`: `date | task | models | rework count | first-pass yes/no | defect`. Skip trivial tasks, since logging them wastes tokens.
3. **Analyze**: when the same defect appears twice, fix the process by editing the matching rule in these skills, not just the instance.
4. **Improve**: change one rule at a time so the effect is attributable.
5. **Control**: aim for a first-pass yield of 90% or better per route. If a route falls below it (say, haiku on a certain kind of task), promote that route one tier.

## Token habits

- `Grep`/`Glob` before `Read`; read with `offset`/`limit`.
- Batch independent tool calls into one turn.
- Do not re-read a file you just edited or paste large outputs back.
- Cap command output (`| tail -30`, quiet flags).
- Final answer: result, what changed, next step. No recap of the work.
- Suggest `/clear` between unrelated tasks, and `/compact` before context gets heavy.
- Reuse existing skills (`simplify`, `code-review`, `engineering:*`, `artifact-design`, `dataviz`) instead of improvising their procedures.
