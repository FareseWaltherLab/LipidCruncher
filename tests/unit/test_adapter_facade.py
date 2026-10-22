"""Pin: StreamlitAdapter stays the one facade the app calls.

The adapter's code is split across modules; every ``StreamlitAdapter.<name>``
the app or the tests use must still resolve on the class.
"""
import re
from pathlib import Path

from app.adapters.streamlit_adapter import StreamlitAdapter

ROOT = Path(__file__).parents[2]


def test_every_name_called_on_the_adapter_exists():
    used = set()
    for folder in ('src', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            used |= set(re.findall(r'\bStreamlitAdapter\.(\w+)', path.read_text()))
    assert used, "no StreamlitAdapter calls found"
    assert sorted(n for n in used if not hasattr(StreamlitAdapter, n)) == []
