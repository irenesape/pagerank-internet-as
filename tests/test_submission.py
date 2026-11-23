"""Tests for the builder of the self-contained submission notebook.

The submission notebook is generated from a Markdown source plus the package
code, so that the hand-in can run without this repository while
``src/pagerank`` stays the single tested implementation. These tests pin the
builder's behaviour on a tiny source written to a temporary directory: how
the source is split into cells, that inlined code is the package source
verbatim, that hand-derived constants arrive unchanged, that two builds are
byte-identical, and that a final build refuses unfinished work.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import nbformat
import pytest

import pagerank.caida
import pagerank.graph
import pagerank.power

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "submission" / "build_notebook.py"


def _load_builder() -> ModuleType:
    """Import ``submission/build_notebook.py`` by path; it is not a package."""
    module_spec = importlib.util.spec_from_file_location("build_notebook", BUILDER)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    # Registered before it runs: @dataclass looks its module up in
    # sys.modules while the class is being created.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


builder = _load_builder()

CONSTANTS = '''\
from fractions import Fraction

EDGES = [(0, 1), (1, 0)]
N = 2
HALF = Fraction(1, 2)
LONG = (Fraction(659, 1769), Fraction(27713, 141520), Fraction(2789, 7076), Fraction(3, 80))
MISSING = None
'''

SOURCE = '''\
# Title

First paragraph.

<!-- cell -->

Second Markdown cell.

```python
import numpy as np
x = np.arange(3)
```

```inline
pagerank.graph:DirectedGraph
pagerank.power:pagerank_power
```

Text between code cells, with a fenced example that stays Markdown:

```text
```python
not a code cell
<!-- cell -->
```

```constants
EDGES
N
HALF
LONG
```
'''


def _write(tmp_path: Path, source: str, constants: str = CONSTANTS) -> tuple[Path, Path]:
    source_path = tmp_path / "source.md"
    constants_path = tmp_path / "constants.py"
    source_path.write_text(source, encoding="utf-8")
    constants_path.write_text(constants, encoding="utf-8")
    return source_path, constants_path


def _build(tmp_path: Path, source: str = SOURCE, **kwargs) -> nbformat.NotebookNode:
    source_path, constants_path = _write(tmp_path, source)
    return builder.build_notebook(source_path, constants_path, **kwargs)


# --------------------------------------------------------------------------
# Splitting the source into cells
# --------------------------------------------------------------------------


def test_source_is_split_into_the_expected_cells(tmp_path: Path) -> None:
    """Markdown, a cell break, then python, inline, Markdown and constants."""
    nb = _build(tmp_path)
    kinds = [cell.cell_type for cell in nb.cells]
    assert kinds == ["markdown", "markdown", "code", "code", "markdown", "code"]
    assert nb.cells[0].source == "# Title\n\nFirst paragraph."
    assert nb.cells[1].source == "Second Markdown cell."


def test_python_block_is_copied_as_is(tmp_path: Path) -> None:
    nb = _build(tmp_path)
    assert nb.cells[2].source == "import numpy as np\nx = np.arange(3)"


def test_other_fences_stay_inside_the_markdown(tmp_path: Path) -> None:
    """A ```text block is Markdown content: no code cell, no split inside it."""
    nb = _build(tmp_path)
    text = nb.cells[4].source
    assert text.startswith("Text between code cells")
    assert "```python\nnot a code cell\n<!-- cell -->\n```" in text


def test_inline_block_is_the_package_source_verbatim(tmp_path: Path) -> None:
    """Each named object's ``inspect.getsource``, in order, two blank lines apart."""
    nb = _build(tmp_path)
    expected = (
        inspect.getsource(pagerank.graph.DirectedGraph).rstrip("\n")
        + "\n\n\n"
        + inspect.getsource(pagerank.power.pagerank_power).rstrip("\n")
    )
    assert nb.cells[3].source == expected


def test_inline_constant_brings_its_documentation_comment() -> None:
    """A module-level constant is inlined from its assignment and ``#:`` lines."""
    source = builder.object_source("pagerank.caida:C2P")
    assert source.splitlines()[-1] == "C2P: Final = -1"
    assert source.startswith("#:")
    namespace: dict = {"Final": object}
    exec(source, namespace)
    assert namespace["C2P"] == pagerank.caida.C2P


def test_inline_refuses_a_name_imported_from_elsewhere() -> None:
    """``pagerank:pagerank_power`` is re-exported, not defined, in ``pagerank``."""
    with pytest.raises(builder.BuildError, match="defined in pagerank.power"):
        builder.object_source("pagerank:pagerank_power")


@pytest.mark.parametrize(
    "target",
    ["pagerank.power", "pagerank.power:no_such_name", "pagerank.nope:x"],
)
def test_inline_refuses_a_bad_target(target: str) -> None:
    with pytest.raises(builder.BuildError):
        builder.object_source(target)


def test_constants_block_writes_values_that_evaluate_back(tmp_path: Path) -> None:
    """The assignments re-create the constants exactly, fractions included."""
    nb = _build(tmp_path)
    code = nb.cells[5].source
    assert code.splitlines()[:3] == [
        "EDGES = [(0, 1), (1, 0)]",
        "N = 2",
        "HALF = Fraction(1, 2)",
    ]
    # LONG does not fit on one line, so it is written one item per line.
    assert "LONG = (\n    Fraction(659, 1769),\n" in code
    namespace: dict = {"Fraction": Fraction}
    exec(code, namespace)
    assert namespace["EDGES"] == [(0, 1), (1, 0)]
    assert namespace["HALF"] == Fraction(1, 2)
    assert namespace["LONG"] == (
        Fraction(659, 1769),
        Fraction(27713, 141520),
        Fraction(2789, 7076),
        Fraction(3, 80),
    )


def test_constants_block_refuses_an_unknown_name(tmp_path: Path) -> None:
    source = "```constants\nNOT_THERE\n```\n"
    with pytest.raises(builder.BuildError, match="NOT_THERE"):
        _build(tmp_path, source)


def test_unclosed_code_block_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(builder.BuildError, match="never closed"):
        _build(tmp_path, "Text.\n\n```python\nx = 1\n")


# --------------------------------------------------------------------------
# Output format and determinism
# --------------------------------------------------------------------------


def test_cells_have_deterministic_ids_and_no_outputs(tmp_path: Path) -> None:
    nb = _build(tmp_path)
    assert [cell.id for cell in nb.cells] == [
        f"cell-{i:03d}" for i in range(len(nb.cells))
    ]
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
    assert nb.metadata.kernelspec.name == "python3"
    nbformat.validate(nb)


def test_two_builds_are_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    builder.write_notebook(_build(tmp_path), first)
    builder.write_notebook(_build(tmp_path), second)
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------
# The final build refuses unfinished work
# --------------------------------------------------------------------------

UNFINISHED = '''\
## Modelling

[AUTHOR: modelling]

```constants
N
MISSING
```
'''


def test_draft_build_accepts_placeholders_and_none(tmp_path: Path) -> None:
    nb = _build(tmp_path, UNFINISHED)
    assert nb.cells[0].source == "## Modelling\n\n[AUTHOR: modelling]"
    assert nb.cells[1].source == "N = 2\nMISSING = None"


def test_final_build_refuses_a_placeholder_and_a_none_constant(tmp_path: Path) -> None:
    with pytest.raises(builder.BuildError) as excinfo:
        _build(tmp_path, UNFINISHED, final=True)
    message = str(excinfo.value)
    assert "[AUTHOR: modelling]" in message
    assert "MISSING is still None" in message


def test_final_build_passes_once_the_work_is_in(tmp_path: Path) -> None:
    finished = "## Modelling\n\nThe author's text.\n\n```constants\nN\n```\n"
    nb = _build(tmp_path, finished, final=True)
    assert len(nb.cells) == 2
