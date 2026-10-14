"""
Base plotter interface.

Defines PlotterServiceProtocol, a structural (duck-typing) protocol that some
plotting services implement; they don't inherit from it. It documents the
interface pattern; no static type checker is configured for the project.
"""

from typing import Dict, List, Protocol, runtime_checkable



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