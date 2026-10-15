"""
Render AI-produced text as markdown without letting it load images.

Streamlit's markdown renders ``![alt](url)`` as an image the browser fetches
on display, with no click. Text the model writes can be steered by the
uploaded data (prompt injection), so an answer could carry an image whose
URL encodes the user's data, and showing the answer would send it to that
host. Escaping the bracket keeps the text readable and nothing loads.
Links are left alone: they need a click and are often useful.
"""
from typing import Iterable, Iterator


def safe_markdown(text: str) -> str:
    """``text`` with every markdown image (inline or reference) inert."""
    return text.replace("![", "!\\[")


def safe_stream(chunks: Iterable[str]) -> Iterator[str]:
    """``safe_markdown`` over a stream whose chunks may split ``![``."""
    held = ""
    for chunk in chunks:
        text = held + chunk
        held = "!" if text.endswith("!") else ""
        yield safe_markdown(text[:-1] if held else text)
    if held:
        yield held
