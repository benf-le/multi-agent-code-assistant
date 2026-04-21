import re
from typing import Iterable


def normalize_marker(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower()).strip('_') or 'marker'


def extract_requirement_lines(brd_content: str) -> list[str]:
    items: list[str] = []
    for raw in brd_content.splitlines():
        line = raw.strip()
        if line.startswith('- '):
            items.append(line[2:].strip())
    return items


def split_into_chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[idx: idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def summarize_text(lines: Iterable[str], max_items: int = 3) -> str:
    selected = list(lines)[:max_items]
    if not selected:
        return 'Feature extracted from BRD.'
    return ' | '.join(selected)
