"""
Base plotter interface.

Defines the PlotterServiceProtocol that all plotting services should follow.
This is a structural (duck-typing) protocol — services don't need to inherit
from it, but they should implement the same interface pattern.

Used for documentation and static type checking (mypy/pyright).
"""

from typing import Dict, List, Protocol, Union, runtime_checkable

import plotly.graph_objects as go


@runtime_checkable
class PlotterServiceProtocol(Protocol):
    """Protocol that plotting services should follow.

    All plotting services in this package share these conventions:

    1. **Static methods only** — no instance state.
    2. **Pure logic** — no Streamlit imports.
    3. **Return figures** — methods return ``go.Figure`` or ``plt.Figure``.
    4. **Validate inputs** — raise ``ValueError`` for invalid inputs.
    5. **Named constants** — no magic numbers in layout code.

    Services may additionally provide:
    - ``generate_color_mapping()`` — consistent color assignment.
    - Data preparation methods returning typed dataclasses.

    This protocol is ``@runtime_checkable`` so tests can verify compliance:

        assert isinstance(MyPlotterService, PlotterServiceProtocol)
    """

    @staticmethod
    def generate_color_mapping(
        items: List[str],
    ) -> Dict[str, str]:
        """Map items (conditions or classes) to colors.

        Args:
            items: List of labels to color.

        Returns:
            Dict mapping label to hex color string.
        """
        ...