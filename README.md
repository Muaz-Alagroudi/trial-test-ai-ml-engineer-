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
this README with one row per query. For each row, report:

- **Retrieved evidence** — the source doc and chunk ID of the top chunk(s) actually
  returned by `retrieve()`, plus a short excerpt of the chunk text (a sentence or two is
  enough). For the unanswerable query, show what *was* retrieved, even if it's irrelevant.
- **Grounded?** — whether the answer was grounded, and what `is_grounded()` returned.
- **Tool call** — whether the model actually requested a tool call, for *every* query (not
  just the NOI one). If it did, give the tool name, the arguments the model passed, and the
  value `calculate_noi()` returned, and say whether the number in the final answer matches
  that value.
- **What the model got wrong** — this is the point: ignored an instruction, formatted
  oddly, tried to answer the NOI itself, called the tool when it shouldn't have, hedged
  when it shouldn't have.

Also commit the raw output of your eval run as `eval/results.json`: for each query, the
retrieved chunks (with IDs), any tool-call messages exchanged with the model, and the
final answer.

A model that behaves perfectly on all 5 is a valid result. Say so explicitly rather than
leaving it implied.

Also fill in the two assertions in `tests/test_permissions.py` proving `retrieve()` never
returns an unauthorized chunk for `user_c` versus `user_a`/`user_b`. Hard requirement, but
deliberately light — not a full test suite.

### Task 3 results

Run setup: `llama3.2` via Ollama for chat, `all-minilm` for embeddings, `top_k=3`, default
sampling, `answer_question()` called directly per query as the listed user. The raw run
(retrieved chunks, every message sent to and returned by the model, final result) is in
`eval/results.json`. The permission tests in `tests/test_permissions.py` pass (2 passed).

| Query | Retrieved evidence (top chunk first) | Grounded? | Tool call | What the model got wrong |
|---|---|---|---|---|
| **q1** user_a: Maple Ridge revenue Q3 2026 | `doc1.txt#0`: "Q3 2026 financial summary for Maple Ridge Apartments: total revenue $412,000, operating expenses $158,000." Also `doc4.txt#0`, `doc6.txt#0`. | Yes, `is_grounded()` = `True`. Answer: revenue was $412,000 [doc1.txt#0]. | Yes, not needed. `calculate_noi(property_id="1", period="Q3-2026")` returned 254,000.00. The answer correctly ignores it and uses $412,000 from the document. | Called the NOI tool on a revenue question. Leaked the internal id ("property_id 1") into the answer instead of the property name. |
| **q2** user_b: Unit 4B HVAC repair | `doc3.txt#0`: "Maintenance log, Maple Ridge Apartments, Unit 4B: HVAC system repaired and inspected in March 2026." Also `doc6.txt#0`, `doc2.txt#0`. | Yes, `True`. Answer: March 2026 [doc3.txt#0]. | Yes, not needed. `calculate_noi(property_id="1", period="March 2026")` returned an error (no data for that period); the error was sent back and the model answered from context. | Called the tool on a maintenance question, with a made-up period. The answer itself is correct and cited. |
| **q3** user_c: office holiday policy 2026 | `doc2.txt#0`: "The office is closed on all federal holidays and the last week of December." Also `doc5.txt#0` (onboarding checklist). user_c only sees the two `all` docs. | Yes, `True`. Answer: closed on federal holidays and the last week of December [doc2.txt#0]. | Yes, not needed. `calculate_noi(property_id="None", period="2026")` returned an error (`'None'` is not an int). | Called the tool on a policy question, with a null id sent as the string `"None"`. The answer is correct and cited. |
| **q4** user_a: Oak Hill Plaza NOI Q3 2026 | `doc4.txt#0`: "Q3 2026 financial summary for Oak Hill Plaza: total revenue $275,000, operating expenses $121,000." Also `doc6.txt#0`, `doc1.txt#0`. | No, `False`: the user saw "I could not get a verified result for this question." | Yes, correct. `calculate_noi(property_id="2", period="Q3-2026")` returned 154,000.00. The final number ($154,000) matches the tool value. | Worked the NOI out itself ("NOI = $275,000 - $121,000 = $154,000") instead of only narrating the tool result, which the task forbids. `shows_own_arithmetic()` caught it and the answer was withheld even though the number is right. Also leaked "property_id 2". |
| **q5** user_b: Unit 12 security deposit (**the unanswerable query**) | `doc6.txt#0`: "Lease renewal notice, Oak Hill Plaza, Unit 12, effective November 2026. Tenant has elected to renew for a 12-month term at the existing rate." Also `doc3.txt#0`, `doc2.txt#0`. user_b may see doc6, but no document states a deposit amount. | Yes, `True`: a correct refusal, cited to [doc6.txt#0]. | Yes, not needed. `calculate_noi(property_id="2", period="Q3-2026")` returned 154,000.00 (Oak Hill's NOI), which the answer correctly ignores. | Declined correctly, but not with the exact "I don't know" it was told to use; it wrote a longer explanation instead. Also called the tool for no reason. |

**The model did not behave perfectly on any of the 5 queries.** The shared failure is that
llama3.2 requests `calculate_noi` on every query, including the four that do not ask for
NOI, often with junk arguments (`"March 2026"`, `"None"`). The pipeline tolerates this:
bad arguments come back as an error message, and a tool result only counts as evidence on
NOI questions, so the NOI figure cannot be passed off as something else.

Also seen in earlier runs with default sampling, so not in `eval/results.json`:

- q1 once reported the tool's NOI ($254,000.00) as the revenue. The grounding check caught
  it and the answer was withheld.
- q3 once answered "I don't know" even though `doc2.txt#0` answers it. In 15 more q3 runs
  that did not recur, but 3 of the 15 returned a fake tool call written as plain text
  (`{"name": "get_holiday_schedule", ...}`), which was marked ungrounded. At temperature 0,
  5 of 5 q3 runs answered correctly.
- q4 once showed the same arithmetic as above, with a label in brackets
  ("$275,000 (total revenue) - $121,000"), which the first version of the arithmetic check
  missed. The pattern now allows that label.

## Task 4 — Design judgment (15 min)

Two or three sentences each, in this README:

- The real deployment runs a ~27B model on a single Mac Studio serving multiple employees
  at once, not a per-request API call from a laptop. What would you change about your
  prompt/context design or retrieval `top_k` if you were memory- or context-window-
  constrained rather than calling a hosted API freely?
- What's the one part of your solution you're least confident is airtight, and why?

### Task 4 answers

**Memory- or context-constrained deployment.** I would add a reranking step: retrieve a
wider candidate set cheaply by embedding similarity, then rerank with a small cross-encoder
and pass only the top 1 or 2 chunks, so fewer and more relevant tokens reach the 27B model.
I would also move the fixed instructions (tool rules, citation format, "I don't know"
rule) out of the per-request user prompt into one system prompt that stays identical across
requests, so the server can cache its KV state once and reuse it for every employee's
request instead of re-processing it each time.

**Least airtight part.** Identifying the claims in an answer. `_extract_claims()` guesses
entities from capitalisation and digits, and `shows_own_arithmetic()` is a regex, so both
depend on how the model happens to phrase and format things: a line break once glued two
claims together, and a label in brackets once hid the arithmetic from the check. A claim
written in lowercase, as words ("two hundred thousand"), or paraphrased past the fuzzy
threshold can slip through or be wrongly rejected; an NER model or an LLM-based claim
extractor would be the next step.


## Submitting

1. Clone this repo and work on it locally, for 2 hours or less.
2. When you're done, make **exactly one commit** containing all of your work.
3. Submit the Google Form you were sent, with a link to your repo. Only one commit is
   allowed — any additional commits made after the time you submit the form will be
   flagged as cheating and will disqualify your submission. Commit once, when you're truly
   done, then submit the form.
