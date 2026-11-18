# agents/: persistent memory for the coding agent

This folder is the agent's long-term memory for vla.cu. Chat context is lost between sessions; this folder is not. It records where the project stands, what was researched, what was tried, what failed, and why decisions were made.

**It is mandatory to read this folder at the start of every session and to update it before every session ends** (and after every chunk, see below). The human can read and edit any file here. If the human corrects something, the correction wins and gets logged.

## Files

| File | What it holds | How it changes |
|---|---|---|
| `STATE.md` | Where we are right now: last chunk done, what works, what's broken, the next suggested options | **Overwritten** each update (keep it short, under ~60 lines) |
| `EXPERIMENTS.md` | Every experiment: change, prediction, result, explanation, verdict (worked / failed / inconclusive) | **Append-only**. Never delete failures |
| `RESEARCH.md` | Things looked up: concepts, library behavior, spec numbers, with source and date | **Append-only**, newest at the bottom; mark outdated entries instead of deleting |
| `DECISIONS.md` | Why we chose X over Y (design choices, tool choices, scope cuts) | **Append-only**; supersede old decisions with a new entry |
| `HARDWARE.md` | The detected GPU, toolchain versions, peak bandwidth with source, measured baselines | Updated when the environment changes; keep a dated change log at the bottom |
| `BACKLOG.md` | Ideas, suggested next steps, open questions, things deferred | Add and tick off; move finished or dropped items to the "Done / dropped" section with a reason |

## Session start protocol (always do this first)

1. Read `agents/STATE.md`, then `HARDWARE.md`.
2. Skim the tail of `EXPERIMENTS.md` (last ~5 entries), and `DECISIONS.md` if the task touches a past choice.
3. Search `EXPERIMENTS.md` and `RESEARCH.md` for the topic at hand **before** re-researching or re-trying anything. If something failed before, say so and cite the entry.
4. Open the session by telling the human, in 3-5 lines: where we left off, what's broken, and the options from `BACKLOG.md` that seem most relevant. Do not start work until the human picks.

## Session end protocol and after every chunk

1. Update `STATE.md` (overwrite): last chunk, what runs, what's failing, next options.
2. Append any experiment to `EXPERIMENTS.md`, including failures and surprises, using the template below.
3. Append anything newly researched to `RESEARCH.md` with source and date.
4. Append any non-trivial choice to `DECISIONS.md`.
5. Update `BACKLOG.md` (new ideas in, finished items out).
6. Tell the human which files in `agents/` were updated and show the diff or a one-line summary per file.

If a session is cut short, a half-updated `STATE.md` that says "interrupted after X" beats no update.

## Rules for writing here

- **Date everything** (YYYY-MM-DD) and, for experiments, record the git commit hash (`git rev-parse --short HEAD`) so results are reproducible.
- **Measured numbers only**, each with the GPU name and the exact command used. Never record a number that was not measured, and label estimates as "estimate".
- **Never delete or rewrite history.** If an entry turns out wrong, add a correction note below it (`CORRECTION YYYY-MM-DD: ...`) and fix the facts in `STATE.md`.
- **Failures are first-class.** A failed experiment with a clear explanation is as valuable as a win. Record what was expected, what happened, and the leading hypothesis for why.
- **Separate fact from guess.** Mark guesses as "hypothesis" and untested claims as "unverified". Promote them once tested.
- **Sources for research:** URL or doc name plus date. If something came from memory instead of a lookup, say "from memory, unverified".
- **Keep the human's reasoning separate.** The human's predictions and questions arrive in chat; record them verbatim in the `Prediction` field of their experiment entry. The agent never invents or rewrites the human's reasoning, and never creates a file that pretends to be their notes.
- **Be concise.** Bullets over paragraphs. If a file gets long, add a short summary at the top rather than deleting content.

## Experiment entry template (`EXPERIMENTS.md`)

```
## EXP-NNN: <short name>  (YYYY-MM-DD, commit <hash>)
- Goal / question:
- Change (one thing):
- Prediction (human's, if given; else agent's, labeled):
- Command:
- GPU / environment:
- Result (measured):
- Explanation / hypothesis:
- Verdict: worked | failed | inconclusive
- Follow-ups:
```

## Research entry template (`RESEARCH.md`)

```
## <topic>  (YYYY-MM-DD)
- Question:
- Finding:
- Source: <url or doc> (or "from memory, unverified")
- Confidence: high | medium | low
- Relevance to vla.cu:
```

## Decision entry template (`DECISIONS.md`)

```
## DEC-NNN: <decision>  (YYYY-MM-DD)
- Context:
- Options considered:
- Chosen and why:
- Revisit if:
- Status: active | superseded by DEC-NNN
```
