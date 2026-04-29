"""Project-level helpers for managing current_project state.

These functions manage the in-memory project dictionary (dict[str, str])
that tracks the cumulative state of the generated codebase across tasks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.agents.serialization_utils import to_plain_dict, to_pretty_json


# ============================================================
# File extraction from DevResult
# ============================================================

def get_file_content_from_dev_result(
    dev_result: Any,
    file_path: str,
) -> str | None:
    """Return file content from a DevResult for the given *file_path*.

    Parameters
    ----------
    dev_result : DevResult (Pydantic) or dict
        The developer output to search.
    file_path : str
        Relative file path to look up.  Comparison is case-insensitive.

    Returns
    -------
    str | None
        The ``code`` field if found, otherwise ``None``.
    """
    data = to_plain_dict(dev_result)
    if not isinstance(data, dict):
        return None

    target = file_path.lower()
    for entry in data.get("files", []):
        if isinstance(entry, dict) and entry.get("file_path", "").lower() == target:
            return entry.get("code")

    return None


# ============================================================
# Project context extraction
# ============================================================

def extract_project_context_from_current_project(
    current_project: dict[str, str],
) -> str:
    """Build a project-context string from README.md and project_context.json.

    Parameters
    ----------
    current_project : dict[str, str]
        Mapping of ``file_path -> content``.

    Returns
    -------
    str
        Formatted context string.  Empty string when neither file exists.
    """
    parts: list[str] = []

    readme = _find_case_insensitive(current_project, "README.md")
    if readme is not None:
        parts.append(f"## README.md\n{readme}")

    ctx_json = _find_case_insensitive(current_project, "project_context.json")
    if ctx_json is not None:
        parts.append(f"## project_context.json\n{ctx_json}")

    return "\n\n".join(parts)


def _find_case_insensitive(mapping: dict[str, str], key: str) -> str | None:
    """Look up *key* in *mapping* with case-insensitive comparison."""
    key_lower = key.lower()
    for k, v in mapping.items():
        if k.lower() == key_lower:
            return v
    return None


# ============================================================
# Foundation file assertions
# ============================================================

def assert_required_foundation_files(current_project: dict[str, str]) -> None:
    """Raise ``ValueError`` if the foundation files are missing.

    TASK-001 must produce both ``README.md`` and ``project_context.json``.
    """
    if _find_case_insensitive(current_project, "README.md") is None:
        raise ValueError(
            "TASK-001 foundation error: README.md is missing from the project. "
            "The foundation task must create README.md as the human-readable project contract."
        )

    if _find_case_insensitive(current_project, "project_context.json") is None:
        raise ValueError(
            "TASK-001 foundation error: project_context.json is missing from the project. "
            "The foundation task must create project_context.json as the machine-readable project contract."
        )


# ============================================================
# Apply DevResult to project
# ============================================================

def apply_dev_result_to_project(
    current_project: dict[str, str],
    dev_result: Any,
) -> dict[str, str]:
    """Create a new project dict by applying a DevResult on top of *current_project*.

    Behaviour:
    - Never mutates the input ``current_project``.
    - Each file in ``dev_result.files`` and ``dev_result.unit_tests`` is written
      into the new dict keyed by ``file_path``.
    - Invalid paths (starting with ``/`` or ``../``) raise ``ValueError``.

    Parameters
    ----------
    current_project : dict[str, str]
        The current committed project state.
    dev_result : DevResult (Pydantic) or dict
        The developer output to apply.

    Returns
    -------
    dict[str, str]
        A **new** dict representing the updated project.
    """
    data = to_plain_dict(dev_result)
    if not isinstance(data, dict):
        data = {}

    updated: dict[str, str] = deepcopy(current_project)

    for entry in data.get("files", []):
        if not isinstance(entry, dict):
            continue
        fp = entry.get("file_path", "")
        code = entry.get("code", "")
        _validate_file_path(fp)
        updated[fp] = code

    for entry in data.get("unit_tests", []):
        if not isinstance(entry, dict):
            continue
        fp = entry.get("file_path", "")
        code = entry.get("code", "")
        _validate_file_path(fp)
        updated[fp] = code

    return updated


def _validate_file_path(file_path: str) -> None:
    """Raise ``ValueError`` for unsafe relative paths."""
    if not file_path or not file_path.strip():
        raise ValueError("file_path cannot be empty")

    cleaned = file_path.replace("\\", "/")
    if cleaned.startswith("/") or cleaned.startswith("../") or "/../" in cleaned:
        raise ValueError(
            f"file_path must be a safe relative path, got: {file_path!r}"
        )


# ============================================================
# Snapshot builder for prompts
# ============================================================

def build_project_snapshot_for_prompt(
    current_project: dict[str, str],
    max_file_chars: int = 12000,
) -> dict[str, str]:
    """Build a prompt-safe snapshot, truncating oversized files.

    - ``README.md`` and ``project_context.json`` are never truncated.
    - Other files exceeding *max_file_chars* are truncated with a clear
      ``... [truncated for prompt]`` marker.

    Parameters
    ----------
    current_project : dict[str, str]
        Mapping of ``file_path -> content``.
    max_file_chars : int
        Maximum characters per file before truncation.

    Returns
    -------
    dict[str, str]
        A new dict suitable for injecting into an LLM prompt.
    """
    protected_names = {"readme.md", "project_context.json"}
    snapshot: dict[str, str] = {}

    for path, content in current_project.items():
        if path.lower() in protected_names:
            snapshot[path] = content
        elif len(content) > max_file_chars:
            snapshot[path] = content[:max_file_chars] + "\n... [truncated for prompt]"
        else:
            snapshot[path] = content

    return snapshot
