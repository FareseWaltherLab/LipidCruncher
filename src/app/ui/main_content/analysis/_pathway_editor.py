"""Pathway layout editor UI controls.

Provides the customization panel for pathway visualization:
class selector, custom nodes, edge management, node positioning,
coordinate grid, and save/load configuration.
"""

import json
from typing import List

import streamlit as st

from app.services.plotting.pathway_viz import (
    ALL_PATHWAY_NODES,
    ALL_PATHWAY_CLASSES,
    DEFAULT_PATHWAY_CLASSES,
)
from app.ui.main_content.analysis._pathway_state import (
    KEY_ACTIVE,
    KEY_CUSTOM_NODES,
    KEY_ADDED_EDGES,
    KEY_REMOVED_EDGES,
    KEY_POS_OVERRIDES,
    KEY_PENDING_CONFIG,
    init_pathway_state,
    get_active_classes,
    get_custom_nodes,
    get_added_edges,
    get_removed_edges,
    get_position_overrides,
    get_current_pos,
    compute_current_edges,
    reset_layout_state,
)


def display_pathway_layout_editor() -> None:
    """Display the pathway layout editing controls."""
    init_pathway_state()

    st.markdown("---")
    st.markdown("#### Customize Pathway Layout")
    st.markdown(
        "Toggle lipid classes, add custom nodes, or modify edges. "
        "Choose a starting point below."
    )
    with st.expander("Layout Options", expanded=False):
        _display_starting_point_toggle()
        _display_class_selector()
        _display_add_custom_node()
        _display_edge_management()
        _display_node_position_editor()
        _display_coordinate_guide()
        _display_save_load_config()


# ═══════════════════════════════════════════════════════════════════════
# Sub-sections
# ═══════════════════════════════════════════════════════════════════════

def _display_starting_point_toggle() -> None:
    """Render preset buttons: Default, All, Scratch."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Default (18 classes)", key='pathway_start_default'):
            reset_layout_state(list(DEFAULT_PATHWAY_CLASSES))
            st.rerun()
    with col2:
        if st.button("All Classes (28)", key='pathway_start_all'):
            reset_layout_state(list(ALL_PATHWAY_CLASSES))
            st.rerun()
    with col3:
        if st.button("Start from Scratch", key='pathway_start_scratch'):
            reset_layout_state([])
            st.rerun()


def _display_class_selector() -> None:
    """Render multiselect for active lipid classes."""
    st.markdown("##### Select Active Classes")
    st.caption(
        "All 28 curated classes are available. "
        "You can also add custom classes below."
    )

    curated_classes = list(ALL_PATHWAY_NODES.keys())
    custom_nodes = get_custom_nodes()
    all_available = curated_classes + [
        c for c in custom_nodes if c not in curated_classes
    ]

    st.multiselect(
        "Classes to display on the pathway:",
        options=all_available,
        key=KEY_ACTIVE,
    )


def _display_add_custom_node() -> None:
    """Render controls to add a custom class node."""
    st.markdown("##### Add Custom Class")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.text_input("Class name", value="", key='pathway_add_node_name')
    with col2:
        st.number_input("X", value=0.0, step=1.0, key='pathway_add_node_x')
    with col3:
        st.number_input("Y", value=0.0, step=1.0, key='pathway_add_node_y')

    def _on_add_node():
        name = st.session_state.get('pathway_add_node_name', '').strip()
        if not name:
            return
        x = st.session_state.get('pathway_add_node_x', 0.0)
        y = st.session_state.get('pathway_add_node_y', 0.0)
        nodes = get_custom_nodes()
        nodes[name] = (x, y)
        st.session_state[KEY_CUSTOM_NODES] = nodes
        current = list(st.session_state.get(KEY_ACTIVE, []))
        if name not in current:
            current.append(name)
        st.session_state[KEY_ACTIVE] = current

    with col4:
        st.markdown("")
        st.markdown("")
        st.button("Add Node", key='pathway_add_node_btn', on_click=_on_add_node)


def _display_edge_management() -> None:
    """Render edge list, remove edge, add edge controls."""
    st.markdown("##### Manage Edges")

    active_set = set(get_active_classes())
    current_edges = compute_current_edges(active_set)

    if current_edges:
        st.markdown(f"**Current edges** ({len(current_edges)}):")
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
                    added = get_added_edges()
                    if edge in added:
                        added.remove(edge)
                        st.session_state[KEY_ADDED_EDGES] = added
                    elif (edge[1], edge[0]) in added:
                        added.remove((edge[1], edge[0]))
                        st.session_state[KEY_ADDED_EDGES] = added
                    else:
                        removed = get_removed_edges()
                        removed.append(edge)
                        st.session_state[KEY_REMOVED_EDGES] = removed
                    st.rerun()
    else:
        st.caption("No edges to display.")

    # Add edge
    st.markdown("**Add an edge:**")
    active_list = get_active_classes()
    if len(active_list) >= 2:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            source = st.selectbox("From", options=active_list, key='pathway_add_edge_source')
        with col2:
            target_options = [c for c in active_list if c != source]
            target = st.selectbox("To", options=target_options, key='pathway_add_edge_target')
        with col3:
            st.markdown("")
            st.markdown("")
            if st.button("Add Edge", key='pathway_add_edge_btn'):
                if source and target and source != target:
                    added = get_added_edges()
                    edge = (source, target)
                    if edge not in added and (target, source) not in added:
                        added.append(edge)
                        st.session_state[KEY_ADDED_EDGES] = added
                        st.rerun()


def _display_node_position_editor() -> None:
    """Render controls to move a node's position."""
    st.markdown("##### Move a Node")

    active_list = get_active_classes()
    if not active_list:
        return

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        move_cls = st.selectbox(
            "Select node:", options=active_list, key='pathway_move_node_select',
        )
    cur_x, cur_y = get_current_pos(move_cls)
    with col2:
        st.number_input("X", value=float(cur_x), step=1.0, key='pathway_move_node_x', format="%.1f")
    with col3:
        st.number_input("Y", value=float(cur_y), step=1.0, key='pathway_move_node_y', format="%.1f")

    def _on_move_node():
        cls = st.session_state.get('pathway_move_node_select', '')
        x = st.session_state.get('pathway_move_node_x', 0.0)
        y = st.session_state.get('pathway_move_node_y', 0.0)
        if cls:
            overrides = get_position_overrides()
            overrides[cls] = (x, y)
            st.session_state[KEY_POS_OVERRIDES] = overrides

    with col4:
        st.markdown("")
        st.markdown("")
        st.button("Move", key='pathway_move_node_btn', on_click=_on_move_node)


def _display_coordinate_guide() -> None:
    """Render coordinate grid toggle."""
    st.markdown("##### Coordinate Guide")
    st.checkbox("Show coordinate grid (for positioning nodes)", key='pathway_show_grid')


def _display_save_load_config() -> None:
    """Render save/load configuration controls."""
    st.markdown("##### Save / Load Configuration")
    st.caption(
        "Download your current pathway layout as a JSON file, "
        "or upload a previously saved configuration."
    )

    col_save, col_load = st.columns(2)
    with col_save:
        config = {
            'active_classes': get_active_classes(),
            'custom_nodes': get_custom_nodes(),
            'added_edges': get_added_edges(),
            'removed_edges': get_removed_edges(),
            'position_overrides': get_position_overrides(),
        }
        st.download_button(
            "Download Configuration",
            data=json.dumps(config, indent=2),
            file_name="pathway_config.json",
            mime="application/json",
            key='pathway_download_config',
        )
    with col_load:
        uploaded = st.file_uploader(
            "Upload Configuration",
            type=['json'],
            key='pathway_upload_config',
            label_visibility='collapsed',
        )
        if uploaded is not None:
            if not st.session_state.get('_pathway_config_applied'):
                try:
                    loaded = json.loads(uploaded.read().decode('utf-8'))
                    if not isinstance(loaded, dict):
                        st.error(
                            "Invalid configuration file: expected a JSON object (dictionary), "
                            f"but got {type(loaded).__name__}."
                        )
                    elif 'active_classes' not in loaded:
                        st.error("Configuration file must contain 'active_classes'.")
                    else:
                        st.session_state[KEY_PENDING_CONFIG] = {
                            'active_classes': list(loaded['active_classes']),
                            'custom_nodes': dict(loaded.get('custom_nodes') or {}),
                            'added_edges': [list(e) for e in (loaded.get('added_edges') or [])],
                            'removed_edges': [list(e) for e in (loaded.get('removed_edges') or [])],
                            'position_overrides': {
                                k: tuple(v) for k, v in
                                (loaded.get('position_overrides') or {}).items()
                            },
                        }
                        st.session_state['_pathway_config_applied'] = True
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse JSON file: {e}")
        else:
            st.session_state.pop('_pathway_config_applied', None)