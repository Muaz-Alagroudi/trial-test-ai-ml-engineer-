"""
Task 2 -- Deterministic calculation, exposed to the model as a real tool.

NOI must be computed here, in code -- never by the language model. But
unlike a hand-rolled keyword router, the MODEL decides when to call this,
via its own native function-calling interface (OpenAI/Anthropic-style
tool schemas, or Ollama's tool support for models like qwen2.5 / llama3.1).
"""

import csv


def load_property_ids(csv_path: str = "data/properties.csv") -> dict:
    """Map each property name in properties.csv to its numeric property_id."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["name"]: int(row["property_id"]) for row in csv.DictReader(f)}


def calculate_noi(property_id: int, period: str, csv_path: str = "data/properties.csv") -> float:
    """Return revenue - operating_expenses for the given property_id and
    period, read directly from data/properties.csv. No model involved.

    Raises ValueError if no row matches, so the caller can report the
    error back to the model instead of returning a made-up number.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["property_id"]) == property_id and row["period"] == period:
                return float(row["revenue"]) - float(row["operating_expenses"])
    raise ValueError(f"no data for property_id={property_id}, period={period!r}")


# A provider-agnostic description of this tool, in JSON-Schema-ish shape.
# Translate this into whatever your chosen model's API actually expects
# (OpenAI/Anthropic tool schemas differ slightly in envelope, not in spirit).
CALCULATE_NOI_TOOL_SCHEMA = {
    "name": "calculate_noi",
    "description": (
        "Compute Net Operating Income (revenue - operating expenses) for a "
        "property and period. Always use this tool for NOI questions -- "
        "never estimate or compute NOI yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "property_id": {"type": "integer", "description": "The property's numeric id"},
            "period": {"type": "string", "description": "e.g. 'Q3-2026'"},
        },
        "required": ["property_id", "period"],
    },
}
