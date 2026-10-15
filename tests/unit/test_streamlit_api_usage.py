"""No deprecated Streamlit arguments in the UI.

Streamlit 1.54 deprecates `use_container_width` (removal announced for
after 2025-12-31) in favour of `width='stretch'` / `width='content'`; every
use logs a deprecation warning, and the next release may reject it."""
import ast
from pathlib import Path

SRC = Path(__file__).parents[2] / "src"


def test_no_use_container_width():
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "use_container_width":
                offenders.append(f"{path.relative_to(SRC)}:{node.value.lineno}")
    assert offenders == []
