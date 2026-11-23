"""Build the self-contained submission notebook from its Markdown source.

The hand-in is a single ``.ipynb`` that has to run on a machine with no copy
of this repository, so it cannot import ``pagerank``. Rather than keep a
second copy of the implementation by hand, this script writes the notebook
from two sources: the narrative in ``pagerank_internet_as.md``, edited by
hand, and the package code in ``src/pagerank``, copied in verbatim with
``inspect.getsource``. The package stays the tested source of truth, and the
test suite fails when the committed notebook no longer matches a fresh build.

Source format:

* Text outside the fenced blocks below becomes Markdown cells. A line holding
  only ``<!-- cell -->`` starts a new Markdown cell. Any other fenced block
  (```` ```text ````, say) stays inside the Markdown.
* A ```` ```python ```` block becomes one code cell, copied as is.
* A ```` ```inline ```` block lists one ``module:name`` per line. The cell is
  the source of each named object, in the order listed, separated by two
  blank lines. Classes and functions come from ``inspect.getsource``; a
  module-level constant comes from its assignment line (and the ``#:``
  comment above it), since ``inspect`` cannot locate the source of an int.
* A ```` ```constants ```` block lists one name per line from
  ``tests/test_toy_graph.py``. The cell assigns each name to the ``repr`` of
  its value, so the hand-derived fractions reach the notebook unchanged.
* A line ``[AUTHOR: <id>]`` marks a section only the author writes.
  ``--final`` refuses to build while one remains, or while a listed constant
  is still ``None``.

Usage::

    .venv/bin/python submission/build_notebook.py [--final]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
from nbformat import NotebookNode

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "submission" / "pagerank_internet_as.md"
OUTPUT = REPO / "submission" / "pagerank_internet_as.ipynb"
CONSTANTS_MODULE = REPO / "tests" / "test_toy_graph.py"

CELL_BREAK = "<!-- cell -->"
CODE_KINDS = ("python", "inline", "constants")
PLACEHOLDER = re.compile(r"^\[AUTHOR: [A-Za-z0-9_-]+\]$")

#: A constant whose one-line form would exceed this is written one item per
#: line, so the cell stays readable without changing its value.
MAX_LINE = 79

#: Written into every build. Fixed rather than taken from the running
#: interpreter, so two builds on different machines are byte-identical.
METADATA: dict[str, Any] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


class BuildError(Exception):
    """The source cannot be turned into a notebook, or not into a final one."""


@dataclass(frozen=True)
class Block:
    """One cell's worth of source.

    Attributes:
        kind: ``"markdown"``, ``"python"``, ``"inline"`` or ``"constants"``.
        text: The block's text, without the fences and without leading or
            trailing blank lines.
        line: 1-based source line of the first line of ``text``, so errors
            can point at the place to edit.
    """

    kind: str
    text: str
    line: int


def _trimmed(numbered: list[tuple[int, str]]) -> tuple[int, str] | None:
    """Drop leading and trailing blank lines; ``None`` if nothing is left."""
    while numbered and not numbered[0][1].strip():
        numbered = numbered[1:]
    while numbered and not numbered[-1][1].strip():
        numbered = numbered[:-1]
    if not numbered:
        return None
    return numbered[0][0], "\n".join(text for _, text in numbered)


def parse_source(text: str) -> list[Block]:
    """Split the Markdown source into blocks, one per notebook cell.

    Args:
        text: The whole source file.

    Returns:
        The blocks in source order. Empty Markdown stretches are dropped.

    Raises:
        BuildError: If a code block is never closed or is empty.
    """
    blocks: list[Block] = []
    markdown: list[tuple[int, str]] = []

    def flush_markdown() -> None:
        trimmed = _trimmed(markdown)
        if trimmed is not None:
            blocks.append(Block("markdown", trimmed[1], trimmed[0]))
        markdown.clear()

    lines = text.splitlines()
    i = 0
    in_markdown_fence = False
    while i < len(lines):
        line = lines[i]
        lineno = i + 1
        opener = line.rstrip()
        kind = opener[3:] if opener.startswith("```") else None

        if in_markdown_fence:
            # Inside a Markdown fence only a bare ``` closes it; everything
            # else, code-block openers and cell breaks included, is content.
            if opener == "```":
                in_markdown_fence = False
            markdown.append((lineno, line))
            i += 1
            continue

        if kind in CODE_KINDS:
            flush_markdown()
            body: list[tuple[int, str]] = []
            i += 1
            while i < len(lines) and lines[i].rstrip() != "```":
                body.append((i + 1, lines[i]))
                i += 1
            if i == len(lines):
                raise BuildError(f"line {lineno}: ```{kind} block is never closed")
            trimmed = _trimmed(body)
            if trimmed is None:
                raise BuildError(f"line {lineno}: ```{kind} block is empty")
            blocks.append(Block(kind, trimmed[1], trimmed[0]))
            i += 1
            continue

        if kind is not None:
            # Any other fence opens a block of Markdown content.
            in_markdown_fence = True
            markdown.append((lineno, line))
        elif line.strip() == CELL_BREAK:
            flush_markdown()
        else:
            markdown.append((lineno, line))
        i += 1

    flush_markdown()
    return blocks


def _entries(block: Block) -> list[tuple[int, str]]:
    """The non-blank lines of a block, stripped, with their source lines."""
    return [
        (block.line + offset, line.strip())
        for offset, line in enumerate(block.text.splitlines())
        if line.strip()
    ]


def _assignment_source(module: ModuleType, name: str) -> str | None:
    """Source lines of a module-level assignment to ``name``, or ``None``.

    Includes the ``#:`` documentation comments directly above it, which is
    where this package documents its constants.
    """
    source = inspect.getsource(module)
    lines = source.splitlines()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            start = node.lineno - 1
            while start > 0 and lines[start - 1].startswith("#:"):
                start -= 1
            return "\n".join(lines[start : node.end_lineno])
    return None


def object_source(entry: str) -> str:
    """Return the verbatim source of the object named ``module:name``.

    The object must be defined in the module named, not merely imported
    there, so that each object can be inlined only once and the notebook
    never carries two copies of the same code.

    Args:
        entry: For example ``"pagerank.power:pagerank_power"``.

    Returns:
        The source text, without a trailing newline.

    Raises:
        BuildError: If the entry is malformed, the module or object does not
            exist, the object is defined elsewhere, or no source is found.
    """
    module_name, sep, name = entry.partition(":")
    if not sep or not module_name or not name or ":" in name:
        raise BuildError(f"{entry!r}: expected 'module:name'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BuildError(f"{entry}: cannot import {module_name} ({exc})") from None
    if not hasattr(module, name):
        raise BuildError(f"{entry}: {module_name} has no attribute {name}")
    obj = getattr(module, name)

    if inspect.isclass(obj) or inspect.isfunction(obj):
        if obj.__module__ != module.__name__:
            raise BuildError(
                f"{entry}: defined in {obj.__module__}, name it from there"
            )
        return inspect.getsource(obj).rstrip("\n")

    source = _assignment_source(module, name)
    if source is None:
        raise BuildError(
            f"{entry}: not a class, a function or a module-level assignment "
            f"in {module_name}"
        )
    return source


def render_inline(block: Block) -> str:
    """The code cell for an ``inline`` block: each object's source, in order."""
    sources = []
    for lineno, entry in _entries(block):
        try:
            sources.append(object_source(entry))
        except BuildError as exc:
            raise BuildError(f"line {lineno}: {exc}") from None
    return "\n\n\n".join(sources)


def load_constants_module(path: Path) -> ModuleType:
    """Import the module holding the hand-derived constants, by file path.

    The test modules are not a package, so they are loaded from their path
    rather than through ``import``.
    """
    module_spec = importlib.util.spec_from_file_location("_notebook_constants", path)
    if module_spec is None or module_spec.loader is None:
        raise BuildError(f"cannot load constants from {path.name}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def format_constant(name: str, value: Any) -> str:
    """``NAME = repr(value)``, one item per line when that is too long.

    Both forms evaluate to the same value; the second only keeps a long tuple
    of fractions readable in the cell.
    """
    text = f"{name} = {value!r}"
    if len(text) <= MAX_LINE or type(value) not in (list, tuple):
        return text
    opening, closing = ("[", "]") if type(value) is list else ("(", ")")
    items = "\n".join(f"    {item!r}," for item in value)
    return f"{name} = {opening}\n{items}\n{closing}"


def build_notebook(
    source_path: Path = SOURCE,
    constants_path: Path = CONSTANTS_MODULE,
    *,
    final: bool = False,
) -> NotebookNode:
    """Build the notebook from ``source_path``.

    Args:
        source_path: The Markdown source.
        constants_path: The module ``constants`` blocks read from.
        final: Refuse to build while an author placeholder remains or a
            listed constant is ``None``.

    Returns:
        A validated nbformat-4 notebook with no outputs, no execution counts
        and cell ids ``cell-000``, ``cell-001``, ... in order.

    Raises:
        BuildError: On a malformed source, or, with ``final``, listing every
            placeholder and ``None`` constant still present.
    """
    blocks = parse_source(source_path.read_text(encoding="utf-8"))
    constants: ModuleType | None = None
    unfinished: list[str] = []
    cells: list[NotebookNode] = []

    for block in blocks:
        if block.kind == "markdown":
            for offset, line in enumerate(block.text.splitlines()):
                if PLACEHOLDER.match(line.strip()):
                    unfinished.append(
                        f"line {block.line + offset}: {line.strip()} is not written"
                    )
            cells.append(nbformat.v4.new_markdown_cell(block.text))
        elif block.kind == "python":
            cells.append(nbformat.v4.new_code_cell(block.text))
        elif block.kind == "inline":
            cells.append(nbformat.v4.new_code_cell(render_inline(block)))
        else:  # constants
            if constants is None:
                constants = load_constants_module(constants_path)
            assignments = []
            for lineno, name in _entries(block):
                if not name.isidentifier() or not hasattr(constants, name):
                    raise BuildError(
                        f"line {lineno}: {name!r} is not a constant of "
                        f"{constants_path.name}"
                    )
                value = getattr(constants, name)
                if value is None:
                    unfinished.append(f"line {lineno}: {name} is still None")
                assignments.append(format_constant(name, value))
            cells.append(nbformat.v4.new_code_cell("\n".join(assignments)))

    if final and unfinished:
        raise BuildError(
            "refusing a final build:\n  " + "\n  ".join(unfinished)
        )

    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=METADATA)
    nbformat.validate(notebook)
    return notebook


def write_notebook(notebook: NotebookNode, path: Path) -> None:
    """Write ``notebook`` to ``path``; same notebook, same bytes."""
    nbformat.write(notebook, str(path))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        description="Build the submission notebook from its Markdown source."
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="refuse to build while an author section or a constant is missing",
    )
    args = parser.parse_args(argv)
    try:
        notebook = build_notebook(final=args.final)
    except BuildError as exc:
        print(f"build_notebook: {exc}", file=sys.stderr)
        return 1
    write_notebook(notebook, OUTPUT)
    n_code = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(
        f"wrote {OUTPUT.relative_to(REPO)}: {len(notebook.cells)} cells, "
        f"{n_code} code"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
