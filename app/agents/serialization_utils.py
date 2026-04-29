"""Shared serialization helpers for agent modules.

Extracted from openai_agents.py to allow project_utils.py (and other
modules) to use them without creating circular imports.
"""

from __future__ import annotations

import json
from typing import Any


def to_plain_dict(obj: Any) -> Any:
    """Recursively convert Pydantic models or lists/dicts of them to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if isinstance(obj, list):
        return [to_plain_dict(item) for item in obj]

    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}

    return obj


def to_pretty_json(data: Any) -> str:
    """Convert data to a pretty-printed JSON string for prompts."""
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    elif isinstance(data, list):
        data = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]
    elif isinstance(data, dict):
        data = {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for key, value in data.items()
        }

    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
