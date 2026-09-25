"""
Tasks 1 & 2 -- Real generation, grounding, and model-driven tool-calling.

Wire this up to an ACTUAL language model that supports native tool/function
calling (a hosted API, or a local model via Ollama such as qwen2.5 or
llama3.1) -- not a stub, and not a hand-rolled keyword router. Which model
you use isn't graded; whether you can actually work with a real model's
real behavior is.
"""

import os
import re
from typing import List, Dict

from ollama import Client
from rapidfuzz import fuzz

from .retrieval import retrieve, Chunk, OLLAMA_BASE_URL
from .tools import calculate_noi, load_property_ids, CALCULATE_NOI_TOOL_SCHEMA

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

_client = Client(host=OLLAMA_BASE_URL)


def _to_ollama_tool(tool: dict) -> dict:
    # Ollama expects OpenAI-style envelopes: {"type": "function", "function": {...}}.
    # The schemas in tools.py are the bare inner object, so wrap them.
    if tool.get("type") == "function":
        return tool
    return {"type": "function", "function": tool}


def call_model(prompt: str | list, tools: list | None = None) -> Dict:
    """Call your chosen real language model.

    Return a normalized dict shaped like:
        {"text": str | None, "tool_call": {"name": str, "arguments": dict} | None}

    - If the model just answers, `text` is set and `tool_call` is None.
    - If the model wants to call a tool (only possible when `tools` is
      passed), `tool_call` is set and `text` may be None.

    You will need to translate this normalized shape to/from whatever your
    provider's actual API expects (OpenAI, Anthropic, and Ollama all differ
    slightly in envelope). That translation is the point of this task.

    `prompt` may also be a full list of Ollama chat messages, which the
    tool-calling round trip needs (assistant tool_calls + "tool" results).
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = _client.chat(
        model=CHAT_MODEL,
        messages=messages,
        tools=[_to_ollama_tool(t) for t in tools] if tools else None,
    )
    message = response.message

    tool_call = None
    if message.tool_calls:
        fn = message.tool_calls[0].function
        tool_call = {"name": fn.name, "arguments": dict(fn.arguments)}

    return {"text": message.content or None, "tool_call": tool_call}


UNKNOWN_ANSWER = "I don't know"
UNVERIFIED_ANSWER = "I could not get a verified result for this question."


def build_prompt(query: str, chunks: list) -> str:
    """Inject the retrieved chunks and the question into one prompt.

    retrieve() returns ["unauthorized"] when the user may see none of the
    chunks; in that case no context is injected and the model is told to
    reply with UNKNOWN_ANSWER only.
    """
    if chunks == ["unauthorized"]:
        return (
            "You have no context available for this question.\n"
            f'Reply with exactly: "{UNKNOWN_ANSWER}" and nothing else.\n\n'
            f"Question: {query}"
        )

    # Replace property names with their ids so the model knows which
    # property_id to pass to calculate_noi. Only the prompt text changes;
    # the chunks themselves are untouched.
    property_ids = load_property_ids()

    def with_ids(text: str) -> str:
        for name, pid in property_ids.items():
            text = text.replace(name, f"property_id {pid}")
        return text

    context = "\n\n".join(f"[{c.chunk_id}]\n{with_ids(c.text)}" for c in chunks)
    return (
        "You answer questions about company documents. You have one tool, "
        "calculate_noi, but most questions do NOT need it.\n\n"
        "Step 1: decide whether to call calculate_noi.\n"
        "- Call it ONLY if the question explicitly asks for NOI or net operating income.\n"
        "- Every other question (revenue, expenses, maintenance, leases, policies, dates, "
        "anything else) is NOT an NOI question: do NOT call any tool, answer from the "
        "context instead.\n"
        "- For an NOI question, you MUST call calculate_noi. Do NOT compute NOI yourself, "
        "even if revenue and operating expenses appear in the context, and do NOT state "
        "any NOI figure that did not come from the calculate_noi result.\n"
        "- Tool arguments: property_id is the property's numeric id and period looks like "
        '"Q3-2026". Never pass chunk IDs, dollar amounts or unit numbers as arguments.\n\n'
        "Step 2: answer using ONLY the context below.\n"
        "- Cite the chunk ID in square brackets (e.g. [doc1.txt#0]) for every fact you use.\n"
        f'- If the context does not contain the answer, reply with exactly: "{UNKNOWN_ANSWER}".\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


# Matches a query asking for NOI, spelled out or abbreviated.
NOI_PATTERN = r"noi|net operating income"

# Query keyword pattern -> the tool that must have been called for the answer
# to count as grounded. This does not route anything: the model still decides
# whether to call a tool, this only checks afterwards that it did.
KEYWORD_TOOLS = {NOI_PATTERN: "calculate_noi"}


def is_grounded(
    answer: str,
    chunks: List[Chunk],
    query: str = "",
    tools_called: List[str] | None = None,
) -> bool:
    """Task 1 grounding check: verify the answer's key claims (names,
    numbers, dates) actually appear in the retrieved chunks it was
    supposedly based on. A simple substring/entity check is enough -- it
    needs to exist, not be sophisticated.

    Names/phrases are fuzzy-matched against the cited chunks so small wording
    differences ("Maple Ridge Apts") still count. Numbers are matched exactly
    after stripping "$" and ",": a fuzzy score would accept $421,000 for
    $412,000, which is exactly the hallucination this check exists to catch.

    If the query contains a keyword from KEYWORD_TOOLS (e.g. "NOI"), the
    answer is only grounded if the model actually called the mapped tool.
    """
    if answer.strip().strip(".") == UNKNOWN_ANSWER:
        # Declining to answer makes no claims, so nothing can be ungrounded.
        return True

    called = tools_called or []
    for keyword, tool in KEYWORD_TOOLS.items():
        if re.search(rf"\b(?:{keyword})\b", query, re.IGNORECASE) and tool not in called:
            return False

    if not chunks or chunks == ["unauthorized"]:
        return False

    # Check against the chunks the answer cites; fall back to all retrieved
    # chunks if it cites none.
    cited_ids = set(re.findall(r"\[([^\]]+#\d+)\]", answer))
    cited = [c for c in chunks if c.chunk_id in cited_ids] or chunks
    # Tool results are never cited by chunk ID, so always check against them.
    cited += [c for c in chunks if c.chunk_id in called and c not in cited]
    context = " ".join(c.text for c in cited).lower()
    context_digits = re.sub(r"[$,]", "", context)

    claims = _extract_claims(re.sub(r"\[[^\]]+#\d+\]", "", answer))
    if not claims:
        return False

    for claim in claims:
        if any(ch.isdigit() for ch in claim):
            for number in re.findall(r"\d[\d,.]*\d|\d", claim):
                if number.replace(",", "") not in context_digits:
                    return False
        if fuzz.partial_ratio(claim.lower(), context) < FUZZY_THRESHOLD:
            return False
    return True


FUZZY_THRESHOLD = 85

# Two numbers joined by a spaced arithmetic operator ("$275,000 - $121,000"),
# optionally with a label after the first ("$275,000 (total revenue) - ...").
# The spaces keep hyphenated periods and terms ("Q3-2026", "12-month") out.
ARITHMETIC_PATTERN = re.compile(r"\d(?:\s*\([^)]*\))?\s+[-−–+*/x×]\s+\$?\d")


def shows_own_arithmetic(answer: str) -> bool:
    """True if the answer does a calculation itself instead of only stating
    a figure, e.g. "NOI = $275,000 - $121,000"."""
    return bool(ARITHMETIC_PATTERN.search(answer))


def _extract_claims(answer: str) -> List[str]:
    """Pull out the key claims: runs of capitalized words and numbers
    ("Maple Ridge Apartments", "Unit 4B", "March 2026", "$412,000").
    The first word of each sentence is skipped unless it contains a digit,
    since sentence-initial capitals ("According", "The") aren't entities.
    A line break also ends a sentence, so lines of a worked calculation
    don't run together into one claim.
    """
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        current: List[str] = []
        for i, raw in enumerate(sentence.split()):
            token = raw.strip(".,;:!?()\"'")
            is_entity = token[:1].isupper() or any(ch.isdigit() for ch in token)
            if i == 0 and not any(ch.isdigit() for ch in token):
                is_entity = False
            # An opening parenthesis starts a new entity ("$154,000 (NOI)").
            if current and raw[:1] == "(":
                claims.append(" ".join(current))
                current = []
            if token and is_entity:
                current.append(token)
            else:
                if current:
                    claims.append(" ".join(current))
                current = []
            # A trailing comma/period/paren ends the entity ("Apartments, Unit 4B").
            if current and raw[-1:] in ",.;:)":
                claims.append(" ".join(current))
                current = []
        if current:
            claims.append(" ".join(current))
    return claims


def answer_question(query: str, user_groups: List[str]) -> Dict:
    """Orchestration for both tasks:

    - Retrieve authorized context for the query via retrieve().
    - Call the model WITH the calculate_noi tool available
      (CALCULATE_NOI_TOOL_SCHEMA) so the model itself decides whether this
      question needs the tool -- do not pre-classify the query yourself
      with keywords/regex and route around the model.
    - If the model requests the tool call, execute calculate_noi() in code,
      send the result back to the model, and get its final narrated answer.
      The model must never be allowed to just state the NOI number itself.
    - If the model answers directly instead, apply is_grounded() to check
      it against the retrieved chunks.
    # Muaz: we can check if query has an answer that needs a tool call,
    # and check the is grounded function to filter
    # the query and get keywords like NOI or similar then have a dict
    # that maps each keyword with its corresponding tool. and check if the model called that tool.
    # if it didnt then it is not grounded. an upgrade would be using an LLM for NER.



    Return a dict shaped like:
        {"answer": str, "citations": [...], "grounded": bool, "used_tool": bool}

    Plus "original_answer": the raw model output. "answer" is that same
    text when grounded, otherwise UNVERIFIED_ANSWER.
    """
    chunks = retrieve(query, user_groups)
    messages = [{"role": "user", "content": build_prompt(query, chunks)}]
    tools_called: List[str] = []
    # Tool results are evidence too: the NOI number in the final answer comes
    # from calculate_noi, not from a document, so it is checked against this.
    evidence = [] if chunks == ["unauthorized"] else list(chunks)
    # Only an NOI question can use the tool's number as evidence. llama3.2
    # also calls the tool on unrelated questions, and then must not be able
    # to pass off the NOI figure as, say, revenue.
    asked_for_noi = re.search(rf"\b(?:{NOI_PATTERN})\b", query, re.IGNORECASE)

    result = call_model(messages, tools=[CALCULATE_NOI_TOOL_SCHEMA])
    tool_call = result["tool_call"]
    if tool_call and tool_call["name"] == "calculate_noi":
        args = tool_call["arguments"]
        tools_called.append("calculate_noi")
        try:
            value = calculate_noi(int(args["property_id"]), str(args["period"]))
            tool_output = f"{value:,.2f}"
            tool_ok = True
        except (KeyError, TypeError, ValueError) as e:
            # The model passed arguments the tool can't use (e.g.
            # property_id="4B"). Report it back instead of crashing, so the
            # model can still answer from the context.
            tool_output = f"error: {e}"
            tool_ok = False

        # Ollama's round trip: the assistant turn that requested the tool,
        # then a "tool" turn carrying the result, then ask again for the
        # narrated answer. Tools are not offered on the second call, so the
        # model has to narrate instead of looping on the tool.
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": tool_call}]})
        messages.append({"role": "tool", "content": tool_output, "tool_name": "calculate_noi"})
        result = call_model(messages)

        if tool_ok and asked_for_noi:
            # Describe what the tool computed, not just the bare number, so
            # "NOI" and the property name in the answer count as backed by it.
            names = {pid: name for name, pid in load_property_ids().items()}
            pid = int(args["property_id"])
            evidence.append(
                Chunk(
                    doc_id="calculate_noi",
                    text=(
                        f"NOI (net operating income) for property_id {pid}, "
                        f"{names.get(pid, '')}, {args['period']}: {tool_output}"
                    ),
                    acl=[],
                    chunk_id="calculate_noi",
                )
            )

    original_answer = result["text"] or ""
    cited_ids = set(re.findall(r"\[([^\]]+#\d+)\]", original_answer))
    citations = [c.chunk_id for c in evidence if c.chunk_id in cited_ids or c.chunk_id in tools_called]

    grounded = is_grounded(original_answer, evidence, query, tools_called)
    if tools_called and tool_ok and asked_for_noi:
        # The tool's number must actually appear in the answer: an answer
        # that never states it (or states a different one) is not grounded.
        # Only for NOI questions: llama3.2 also calls the tool on unrelated
        # questions, and those answers shouldn't have to repeat its number.
        answer_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", original_answer)
        if not any(float(n.replace(",", "")) == value for n in answer_numbers):
            grounded = False
    if asked_for_noi and shows_own_arithmetic(original_answer):
        # The model must only narrate the tool's number. Working it out from
        # revenue and expenses, even to the right figure, breaks that rule.
        grounded = False
    # Never show an unverified answer, e.g. an NOI figure the model produced
    # without calling calculate_noi. original_answer keeps the raw model
    # output for the eval report.
    answer = original_answer if grounded else UNVERIFIED_ANSWER

    return {
        "answer": answer,
        "original_answer": original_answer,
        "citations": citations,
        "grounded": grounded,
        "used_tool": bool(tools_called),
    }
