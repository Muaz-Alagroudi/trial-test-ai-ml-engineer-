"""
Tasks 1 & 2 -- Real generation, grounding, and model-driven tool-calling.

Wire this up to an ACTUAL language model that supports native tool/function
calling (a hosted API, or a local model via Ollama such as qwen2.5 or
llama3.1) -- not a stub, and not a hand-rolled keyword router. Which model
you use isn't graded; whether you can actually work with a real model's
real behavior is.
"""

from typing import List, Dict

from .retrieval import retrieve, Chunk
from .tools import calculate_noi, CALCULATE_NOI_TOOL_SCHEMA


def call_model(prompt: str, tools: list | None = None) -> Dict:
    """Call your chosen real language model.

    Return a normalized dict shaped like:
        {"text": str | None, "tool_call": {"name": str, "arguments": dict} | None}

    - If the model just answers, `text` is set and `tool_call` is None.
    - If the model wants to call a tool (only possible when `tools` is
      passed), `tool_call` is set and `text` may be None.

    You will need to translate this normalized shape to/from whatever your
    provider's actual API expects (OpenAI, Anthropic, and Ollama all differ
    slightly in envelope). That translation is the point of this task.
    """
    raise NotImplementedError("Wire up a real model, including its tool-calling protocol")


def is_grounded(answer: str, chunks: List[Chunk]) -> bool:
    """Task 1 grounding check: verify the answer's key claims (names,
    numbers, dates) actually appear in the retrieved chunks it was
    supposedly based on. A simple substring/entity check is enough -- it
    needs to exist, not be sophisticated.
    """
    raise NotImplementedError("Implement a grounding / hallucination check")


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

    Return a dict shaped like:
        {"answer": str, "citations": [...], "grounded": bool, "used_tool": bool}
    """
    raise NotImplementedError("Implement the full answer_question orchestration")
