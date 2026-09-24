# Technical Trial Test: AI/ML Engineer 

## Setup (10 min)

- `documents/` — six short text files. `documents/manifest.json` maps each file to its ACL
  (access-control list). A doc tagged `"all"` means every authenticated user can see it,
  regardless of their own groups.
- `data/properties.csv` — the structured financial data for the tool-call task.
- `eval/queries.json` — five test queries for Task 3 (see below).
- `app/` — a starter package with function signatures and docstrings for everything you
  need to implement (`retrieval.py`, `tools.py`, `generate.py`, `users.py`), plus
  `web.py` — a small, ungraded FastAPI UI wired to `answer_question()` so you have a
  quick way to try queries by hand instead of writing your own throwaway script.
- `tests/test_permissions.py` — a couple of skeleton assertions for Task 3.
- `Dockerfile` / `Makefile` — **the only local requirement is Docker.** `make up` builds
  the image (installing `requirements.txt` inside the container, not on your machine) and
  starts the UI at `http://localhost:5000`. `make test` runs the pytest suite the same way.
  Your working directory is mounted into the container, so edits you make locally take
  effect immediately (`--reload` is on) without a rebuild — a rebuild only happens
  automatically when you change `requirements.txt`.

**A working language model that supports real tool/function-calling — your choice, not
ours.** Task 2 requires the model's native function-calling interface, so pick something
that supports it: a hosted API (OpenAI- or Anthropic-style tool schemas both work), or a
local model via Ollama that supports tools (e.g. `qwen2.5`, `llama3.1`, `mistral-nemo`).

- **Hosted API key:** drop it in a `.env` file in the repo root (e.g. `OPENAI_API_KEY=...`)
  — it's gitignored, and the Makefile picks it up automatically and passes it into the
  container. No code changes needed to wire it through.
- **Local model via Ollama running on your host machine:** from inside the container,
  reach it at `http://host.docker.internal:11434` instead of `localhost:11434` (Docker's
  bridge network can't see `localhost` on your host). On Linux, if that doesn't resolve,
  add `--network host` to the `docker run` line in the Makefile as a workaround.

Which model or where it runs isn't graded — the project's own architecture treats the model
as swappable behind an internal API, so this mirrors that. What's graded is whether you can
actually get retrieval, generation, and tool-calling working against a real model's real
behavior.

**Test users** (`app/users.py`): `user_a` (groups: `finance`), `user_b` (groups: `propteam`),
`user_c` (groups: `all` only, no team).

## Task 1 — Grounded RAG: retrieval + real generation (45 min)

Implement `retrieve()` (`app/retrieval.py`) and the RAG path of `answer_question()`
(`app/generate.py`) so a user's question actually gets answered by a real model, grounded
in retrieved evidence:

- Chunk the six documents, embed each chunk with a real embedding model, and attach ACL
  metadata at ingestion time.
- `retrieve(query, user_groups)` returns only chunks the user is authorized to see (or
  tagged `all`), ranked by embedding similarity. Unauthorized chunks must never be
  constructible from the result — filtering happens inside `retrieve()`, not after.
- Construct a prompt from the retrieved chunks and the question, and get a real generated
  answer from your chosen model — not a stub string.
- Implement `is_grounded()`: verify the answer's key claims actually appear in the chunks
  it cites, and flag when they don't.

Two things are graded together here, not separately: whether retrieval actually finds the
right chunk, and whether the model's answer is honestly grounded in what was retrieved.

## Task 2 — Tool use through the model's real function-calling interface (30 min)

Qwen (in the real project) must never compute something like NOI itself — deterministic
tools do the math, the model only narrates it. Implement this using your model's actual
tool-calling / function-calling API — not a hand-rolled keyword check that routes around
the model entirely.

- `calculate_noi()` (`app/tools.py`) computes NOI directly from `data/properties.csv`. A
  provider-agnostic tool schema is already sketched as `CALCULATE_NOI_TOOL_SCHEMA` — adapt
  it to your provider's actual format.
- Send the user's question to the model with that tool available. When the model requests
  the tool call, execute `calculate_noi()` in code, and return the result to the model so
  it produces a final narrated answer.
- The model must never be allowed to just answer the number itself — if it tries to, that's
  the failure this task exists to catch.

This is deliberately harder than routing with an if-statement: it requires actually
understanding your model's tool-calling protocol, including the back-and-forth message
format most APIs require.

## Task 3 — Run it, and report what the model actually did (30 min)

Run the 5 queries in `eval/queries.json` (one is deliberately unanswerable) through your
pipeline as each listed user — using `make up` and the provided UI, or by calling
`answer_question()` directly, whichever is faster for you — and produce a short table in
this README: query, whether the
right chunk was retrieved, whether the answer was grounded, whether tool-calling worked
correctly for the NOI query, and — this is the point — what the real model actually got
wrong, if anything (ignored an instruction, formatted oddly, tried to answer the NOI itself,
hedged when it shouldn't have). A model that behaves perfectly on all 5 is a valid result —
say so explicitly rather than leaving it implied.

Also fill in the two assertions in `tests/test_permissions.py` proving `retrieve()` never
returns an unauthorized chunk for `user_c` versus `user_a`/`user_b`. Hard requirement, but
deliberately light — not a full test suite.

## Task 4 — Design judgment (15 min)

Two or three sentences each, in this README:

- The real deployment runs a ~27B model on a single Mac Studio serving multiple employees
  at once, not a per-request API call from a laptop. What would you change about your
  prompt/context design or retrieval `top_k` if you were memory- or context-window-
  constrained rather than calling a hosted API freely?
- What's the one part of your solution you're least confident is airtight, and why?


## Submitting

1. Clone this repo and work on it locally, for 2 hours or less.
2. When you're done, make **exactly one commit** containing all of your work.
3. Submit the Google Form you were sent, with a link to your repo. Only one commit is
   allowed — any additional commits made after the time you submit the form will be
   flagged as cheating and will disqualify your submission. Commit once, when you're truly
   done, then submit the form.
