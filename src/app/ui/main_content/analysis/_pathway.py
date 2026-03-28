"""Feature 5: Pathway Visualization analysis with editable layout."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.services.plotting.pathway_viz import (
    ALL_PATHWAY_NODES,
    ALL_PATHWAY_EDGES,
    DEFAULT_PATHWAY_CLASSES,
    PathwayVizPlotterService,
)
from app.ui.download_utils import plotly_svg_download_button, csv_download_button

from app.ui.main_content.analysis._utils import _check_fa_compatibility


# ═══════════════════════════════════════════════════════════════════════
# Session state helpers
# ═══════════════════════════════════════════════════════════════════════

_KEY_ACTIVE = 'pathway_class_selector'
_KEY_CUSTOM_NODES = 'analysis_pathway_custom_nodes'
_KEY_ADDED_EDGES = 'analysis_pathway_added_edges'
_KEY_REMOVED_EDGES = 'analysis_pathway_removed_edges'
_KEY_POS_OVERRIDES = 'analysis_pathway_position_overrides'


def _init_pathway_state() -> None:
    """Ensure pathway session state keys exist with defaults."""
    if _KEY_ACTIVE not in st.session_state:
        st.session_state[_KEY_ACTIVE] = list(DEFAULT_PATHWAY_CLASSES)
    if _KEY_CUSTOM_NODES not in st.session_state:
        st.session_state[_KEY_CUSTOM_NODES] = {}
    if _KEY_ADDED_EDGES not in st.session_state:
        st.session_state[_KEY_ADDED_EDGES] = []
    if _KEY_REMOVED_EDGES not in st.session_state:
        st.session_state[_KEY_REMOVED_EDGES] = []
    if _KEY_POS_OVERRIDES not in st.session_state:
        st.session_state[_KEY_POS_OVERRIDES] = {}


def _get_active_classes() -> list:
    return list(st.session_state.get(_KEY_ACTIVE, DEFAULT_PATHWAY_CLASSES))


def _get_custom_nodes() -> dict:
    return dict(st.session_state.get(_KEY_CUSTOM_NODES) or {})


def _get_added_edges() -> list:
    return list(st.session_state.get(_KEY_ADDED_EDGES) or [])


def _get_position_overrides() -> dict:
    return dict(st.session_state.get(_KEY_POS_OVERRIDES) or {})


def _get_removed_edges() -> list:
    return list(st.session_state.get(_KEY_REMOVED_EDGES) or [])


def _compute_current_edges(active_set: set) -> list:
    """Compute the effective edge list for the current active classes."""
    removed_set = set()
    for a, b in _get_removed_edges():
        removed_set.add((a, b))
        removed_set.add((b, a))

    edges = []
    for a, b in ALL_PATHWAY_EDGES:
        if a in active_set and b in active_set:
            if (a, b) not in removed_set and (b, a) not in removed_set:
                edges.append((a, b))
    for a, b in _get_added_edges():
        if a in active_set and b in active_set:
            edges.append((a, b))
    return edges


# ═══════════════════════════════════════════════════════════════════════
# Layout editor
# ═══════════════════════════════════════════════════════════════════════


def _display_pathway_layout_editor() -> None:
    """Display the pathway layout editing controls."""
    _init_pathway_state()

    with st.expander("Customize Pathway Layout", expanded=False):
        st.markdown(
            "Toggle lipid classes, add custom nodes, or modify edges. "
            "Choose a starting point below."
        )

        # --- Starting point toggle ---
        col_mode1, col_mode2 = st.columns(2)
        with col_mode1:
            if st.button(
                "Start from Default (18 classes)",
                key='pathway_start_default',
            ):
                st.session_state[_KEY_ACTIVE] = list(DEFAULT_PATHWAY_CLASSES)
                st.session_state[_KEY_CUSTOM_NODES] = {}
                st.session_state[_KEY_ADDED_EDGES] = []
                st.session_state[_KEY_REMOVED_EDGES] = []
                st.session_state[_KEY_POS_OVERRIDES] = {}
                for key in ['pathway_edge_to_remove_select',
                            'pathway_add_edge_source',
                            'pathway_add_edge_target',
                            'pathway_move_node_select',
                            'pathway_move_node_x',
                            'pathway_move_node_y']:
                    st.session_state.pop(key, None)
                st.rerun()
        with col_mode2:
            if st.button(
                "Start from Scratch",
                key='pathway_start_scratch',
            ):
                st.session_state[_KEY_ACTIVE] = []
                st.session_state[_KEY_CUSTOM_NODES] = {}
                st.session_state[_KEY_ADDED_EDGES] = []
                st.session_state[_KEY_REMOVED_EDGES] = []
                st.session_state[_KEY_POS_OVERRIDES] = {}
                for key in ['pathway_edge_to_remove_select',
                            'pathway_add_edge_source',
                            'pathway_add_edge_target',
                            'pathway_move_node_select',
                            'pathway_move_node_x',
                            'pathway_move_node_y']:
                    st.session_state.pop(key, None)
                st.rerun()

        # --- Class selector ---
        st.markdown("##### Select Active Classes")
        st.caption(
            "All 28 curated classes are available. "
            "You can also add custom classes below."
        )

        curated_classes = list(ALL_PATHWAY_NODES.keys())
        custom_nodes = _get_custom_nodes()
        all_available = curated_classes + [
            c for c in custom_nodes if c not in curated_classes
        ]

        st.multiselect(
            "Classes to display on the pathway:",
            options=all_available,
            key=_KEY_ACTIVE,
        )

        # --- Add custom node ---
        st.markdown("##### Add Custom Class")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            new_name = st.text_input(
                "Class name", value="", key='pathway_add_node_name',
            )
        with col2:
            new_x = st.number_input(
                "X", value=0.0, step=1.0, key='pathway_add_node_x',
            )
        with col3:
            new_y = st.number_input(
                "Y", value=0.0, step=1.0, key='pathway_add_node_y',
            )
        def _on_add_node():
            """Callback: runs before widgets are instantiated on rerun."""
            name = st.session_state.get('pathway_add_node_name', '').strip()
            if not name:
                return
            x = st.session_state.get('pathway_add_node_x', 0.0)
            y = st.session_state.get('pathway_add_node_y', 0.0)
            nodes = _get_custom_nodes()
            nodes[name] = (x, y)
            st.session_state[_KEY_CUSTOM_NODES] = nodes
            current = list(st.session_state.get(_KEY_ACTIVE, []))
            if name not in current:
                current.append(name)
            st.session_state[_KEY_ACTIVE] = current

        with col4:
            st.markdown("")
            st.markdown("")
            st.button(
                "Add Node", key='pathway_add_node_btn',
                on_click=_on_add_node,
            )

        # --- Edge management ---
        st.markdown("##### Manage Edges")

        active_set = set(_get_active_classes())
        current_edges = _compute_current_edges(active_set)

        if current_edges:
            st.markdown(
                f"**Current edges** ({len(current_edges)}):"
            )
            edge_labels = [f"{a} — {b}" for a, b in current_edges]
            st.caption(", ".join(edge_labels))

        # Remove edge
        st.markdown("**Remove an edge:**")
        if current_edges:
            remove_options = [f"{a} — {b}" for a, b in current_edges]
            col1, col2 = st.columns([3, 1])
            with col1:
                edge_to_remove = st.selectbox(
                    "Select edge to remove:",
                    options=remove_options,
                    key='pathway_edge_to_remove_select',
                    label_visibility='collapsed',
                )
            with col2:
                if st.button("Remove Edge", key='pathway_remove_edge_btn'):
                    if edge_to_remove:
                        parts = edge_to_remove.split(' — ')
                        edge = (parts[0], parts[1])
                        # If it's a user-added edge, remove from added list
                        added = _get_added_edges()
                        if edge in added:
                            added.remove(edge)
                            st.session_state[_KEY_ADDED_EDGES] = added
                        elif (edge[1], edge[0]) in added:
                            added.remove((edge[1], edge[0]))
                            st.session_state[_KEY_ADDED_EDGES] = added
                        else:
                            # It's a default edge — add to removed list
                            removed = _get_removed_edges()
                            removed.append(edge)
                            st.session_state[_KEY_REMOVED_EDGES] = removed
                        st.rerun()
        else:
            st.caption("No edges to display.")

        # Add edge
        st.markdown("**Add an edge:**")
        active_list = _get_active_classes()
        if len(active_list) >= 2:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                source = st.selectbox(
                    "From", options=active_list,
                    key='pathway_add_edge_source',
                )
            with col2:
                target_options = [c for c in active_list if c != source]
                target = st.selectbox(
                    "To", options=target_options,
                    key='pathway_add_edge_target',
                )
            with col3:
                st.markdown("")
                st.markdown("")
                if st.button("Add Edge", key='pathway_add_edge_btn'):
                    if source and target and source != target:
                        added = _get_added_edges()
                        edge = (source, target)
                        if edge not in added and (target, source) not in added:
                            added.append(edge)
                            st.session_state[_KEY_ADDED_EDGES] = added
                            st.rerun()

        # --- Node position editor ---
        st.markdown("##### Move a Node")

        active_list = _get_active_classes()
        if active_list:
            def _get_current_pos(cls_name: str):
                """Get the current (x, y) for a node, respecting overrides."""
                overrides = _get_position_overrides()
                if cls_name in overrides:
                    return overrides[cls_name]
                if cls_name in ALL_PATHWAY_NODES:
                    return ALL_PATHWAY_NODES[cls_name][0], ALL_PATHWAY_NODES[cls_name][1]
                custom = _get_custom_nodes()
                if cls_name in custom:
                    return custom[cls_name]
                return 0.0, 0.0

            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                move_cls = st.selectbox(
                    "Select node:", options=active_list,
                    key='pathway_move_node_select',
                )
            cur_x, cur_y = _get_current_pos(move_cls)
            with col2:
                move_x = st.number_input(
                    "X", value=float(cur_x), step=1.0,
                    key='pathway_move_node_x',
                    format="%.1f",
                )
            with col3:
                move_y = st.number_input(
                    "Y", value=float(cur_y), step=1.0,
                    key='pathway_move_node_y',
                    format="%.1f",
                )

            def _on_move_node():
                cls = st.session_state.get('pathway_move_node_select', '')
                x = st.session_state.get('pathway_move_node_x', 0.0)
                y = st.session_state.get('pathway_move_node_y', 0.0)
                if cls:
                    overrides = _get_position_overrides()
                    overrides[cls] = (x, y)
                    st.session_state[_KEY_POS_OVERRIDES] = overrides

            with col4:
                st.markdown("")
                st.markdown("")
                st.button(
                    "Move", key='pathway_move_node_btn',
                    on_click=_on_move_node,
                )

        # --- Grid toggle ---
        st.markdown("##### Coordinate Guide")
        st.checkbox(
            "Show coordinate grid (for positioning nodes)",
            key='pathway_show_grid',
        )


# ═══════════════════════════════════════════════════════════════════════
# Main display
# ═══════════════════════════════════════════════════════════════════════


def _display_pathway_viz(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display lipid pathway visualization."""
    _init_pathway_state()

    with st.expander(
        "Class Level Breakdown - Pathway Visualization", expanded=True
    ):
        st.markdown(
            "Visualize lipid class relationships on a metabolic pathway diagram."
        )

        st.markdown(
            "**Fold Change** (determines circle size, log2-scaled):\n\n"
            "> Fold Change = Mean(Experimental) / Mean(Control)"
        )
        st.markdown(
            "**Saturation Ratio** (determines circle color, scaled 0 to max):\n\n"
            "> Saturation Ratio = Saturated Chains / Total Chains"
        )
        st.markdown(
            "Classes present in the dataset are shown as filled circles. "
            "Classes on the pathway but absent from your data are shown "
            "as dashed gray outlines."
        )

        _check_fa_compatibility(df)

        valid_conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
        if len(valid_conditions) < 2:
            st.warning(
                "Pathway visualization requires at least 2 conditions "
                "with multiple samples."
            )
            return

        st.markdown("---")
        st.markdown("#### Data Selection")

        col1, col2 = st.columns(2)
        with col1:
            control = st.selectbox(
                "Control Condition",
                valid_conditions,
                index=0,
                key='pathway_control',
            )
        with col2:
            exp_options = [c for c in valid_conditions if c != control]
            experimental = st.selectbox(
                "Experimental Condition",
                exp_options,
                index=0,
                key='pathway_experimental',
            )

        if control == experimental:
            st.warning("Control and experimental conditions must be different.")
            return

        # --- Layout editor ---
        _display_pathway_layout_editor()

        # --- Compute data (cached) ---
        data = StreamlitAdapter.compute_pathway_data(
            df, experiment, control, experimental,
        )

        # --- Render figure (uses current layout, not cached) ---
        active_classes = _get_active_classes()
        custom_nodes = _get_custom_nodes()
        added_edges = _get_added_edges()
        removed_edges = _get_removed_edges()
        position_overrides = _get_position_overrides()
        show_grid = st.session_state.get('pathway_show_grid', False)

        figure, pathway_dict = PathwayVizPlotterService.create_pathway_viz(
            data.fold_change_df,
            data.saturation_df,
            active_classes=active_classes,
            custom_nodes=custom_nodes if custom_nodes else None,
            added_edges=added_edges if added_edges else None,
            removed_edges=removed_edges if removed_edges else None,
            position_overrides=position_overrides if position_overrides else None,
            show_grid=show_grid,
        )

        st.markdown("---")
        st.markdown("#### Results")

        if figure is None:
            st.warning("Could not generate pathway visualization.")
            return

        st.plotly_chart(figure, use_container_width=True)
        st.session_state.analysis_pathway_fig = figure
        st.session_state.analysis_all_plots['pathway'] = figure

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                figure,
                f"pathway_visualization_{control}_vs_{experimental}.svg",
                key="analysis_svg_pathway",
            )
        with col2:
            summary_rows = []
            if pathway_dict and 'class' in pathway_dict:
                for i, cls in enumerate(pathway_dict['class']):
                    summary_rows.append({
                        'Lipid Class': cls,
                        'Fold Change': pathway_dict['abundance ratio'][i],
                        'Saturation Ratio': pathway_dict['saturated fatty acids ratio'][i],
                    })
            summary_df = pd.DataFrame(summary_rows)
            csv_download_button(
                summary_df,
                "pathway_visualization_data.csv",
                key="analysis_csv_pathway",
            )

        st.markdown(
            f"**Data Summary:** Comparing {experimental} to {control}"
        )
        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True
            )
