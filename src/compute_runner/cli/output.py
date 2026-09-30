"""Where a command's result goes: one JSON document with --json, otherwise tables and text."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence

import typer
from rich.console import Console, RenderableType
from rich.table import Column, Table
from rich.text import Text

from ..security import redacted_env_record


def job_dict(job) -> dict:
    """A job record for display, with environment values hidden."""
    return redacted_env_record(job.model_dump(mode="json")) | {"url": job.url, "remote_ref": job.remote_ref}


class Output:
    def __init__(self, json_mode: bool, console: Console | None = None):
        self.json = json_mode
        # Created per invocation, so the terminal's current width (COLUMNS) applies.
        self.console = console or Console()

    def emit(self, value) -> None:
        """A result as JSON: compact with --json, highlighted otherwise. Job records are redacted."""
        if hasattr(value, "model_dump"):
            value = job_dict(value)
        text = json.dumps(value, ensure_ascii=False)
        if self.json:
            typer.echo(text)
        else:
            self.console.print_json(text)

    def show(self, *renderables: RenderableType) -> None:
        for renderable in renderables:
            self.console.print(renderable)

    def line(self, text: str, *, style: str | None = None) -> None:
        """Plain text, printed as written: brackets in job names or errors are not markup."""
        self.console.print(text, style=style, markup=False, highlight=False)

    def notes(self, notes: Iterable[str]) -> None:
        for note in notes:
            self.line(note, style="dim")


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    wrap: Sequence[str] = (),
    total: Sequence[str] | None = None,
) -> Table:
    """A table of plain-text cells, so nothing in them is read as markup.

    Only the columns named in wrap give way on a narrow terminal; total is a bold last row.
    """
    result = Table(*(Column(header, no_wrap=header not in wrap) for header in headers))
    for row in rows:
        result.add_row(*map(Text, row))
    if total is not None:
        result.add_section()
        result.add_row(*map(Text, total), style="bold")
    return result


def duration(seconds: float | None) -> str:
    """Compact time such as 45s, 12m, 3h 05m or 2d 4h; blank when unknown."""
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"
    hours = seconds // 3600
    return f"{hours}h {seconds // 60 % 60:02d}m" if hours < 48 else f"{hours // 24}d {hours % 24}h"


def ago(seconds: float | None) -> str:
    return "never" if seconds is None else f"{duration(seconds)} ago"


def until(timestamp: float | None) -> str:
    """When a future moment comes, such as "in 2d 4h"; blank when unknown."""
    return "" if timestamp is None else f"in {duration(timestamp - time.time())}"


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def size(value: int | None) -> str:
    """Bytes as B, KB, MB, GB or TB; blank when unknown."""
    if value is None:
        return ""
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"
