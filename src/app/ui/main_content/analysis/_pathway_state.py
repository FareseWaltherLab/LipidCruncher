"""Pathway visualization session state helpers and edge computation.

Manages pathway layout state (active classes, custom nodes, edges,
position overrides) in Streamlit session state.
"""

from typing import Dict, List, Set, Tuple

import streamlit as st

from app.services.plotting.pathway_viz import (
    ALL_PATHWAY_NODES,
    ALL_PATHWAY_EDGES,
    DEFAULT_PATHWAY_CLASSES,
)


# ═══════════════════════════════════════════════════════════════════════
# Session state keys
# ═══════════════════════════════════════════════════════════════════════

KEY_ACTIVE = 'pathway_class_selector'
KEY_CUSTOM_NODES = 'analysis_pathway_custom_nodes'
KEY_ADDED_EDGES = 'analysis_pathway_added_edges'
KEY_REMOVED_EDGES = 'analysis_pathway_removed_edges'
KEY_POS_OVERRIDES = 'analysis_pathway_position_overrides'

KEY_PENDING_CONFIG = '_pathway_pending_config'


# ═══════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════

def init_pathway_state() -> None:
    """Ensure pathway session state keys exist with defaults.

    Also applies any pending uploaded configuration (must happen before
    widgets that are bound to these keys are instantiated).
    """
    # Apply pending uploaded config before widgets bind to the keys.
    pending = st.session_state.pop(KEY_PENDING_CONFIG, None)
    if pending is not None:
        st.session_state[KEY_ACTIVE] = pending['active_classes']
        st.session_state[KEY_CUSTOM_NODES] = pending.get('custom_nodes', {})
        st.session_state[KEY_ADDED_EDGES] = pending.get('added_edges', [])
        st.session_state[KEY_REMOVED_EDGES] = pending.get('removed_edges', [])
        st.session_state[KEY_POS_OVERRIDES] = pending.get('position_overrides', {})

    if KEY_ACTIVE not in st.session_state:
        st.session_state[KEY_ACTIVE] = list(DEFAULT_PATHWAY_CLASSES)
    if KEY_CUSTOM_NODES not in st.session_state:
        st.session_state[KEY_CUSTOM_NODES] = {}
    if KEY_ADDED_EDGES not in st.session_state:
        st.session_state[KEY_ADDED_EDGES] = []
    if KEY_REMOVED_EDGES not in st.session_state:
        st.session_state[KEY_REMOVED_EDGES] = []
    if KEY_POS_OVERRIDES not in st.session_state:
        st.session_state[KEY_POS_OVERRIDES] = {}


# ═══════════════════════════════════════════════════════════════════════
# Accessors
# ═══════════════════════════════════════════════════════════════════════

def get_active_classes() -> List[str]:
    return list(st.session_state.get(KEY_ACTIVE, DEFAULT_PATHWAY_CLASSES))


def get_custom_nodes() -> Dict[str, Tuple[float, float]]:
    return dict(st.session_state.get(KEY_CUSTOM_NODES) or {})


def get_added_edges() -> List[Tuple[str, str]]:
    return list(st.session_state.get(KEY_ADDED_EDGES) or [])


def get_removed_edges() -> List[Tuple[str, str]]:
    return list(st.session_state.get(KEY_REMOVED_EDGES) or [])


def get_position_overrides() -> Dict[str, Tuple[float, float]]:
    return dict(st.session_state.get(KEY_POS_OVERRIDES) or {})


def get_current_pos(cls_name: str) -> Tuple[float, float]:
    """Get the current (x, y) for a node, respecting overrides."""
    overrides = get_position_overrides()
    if cls_name in overrides:
        return overrides[cls_name]
    if cls_name in ALL_PATHWAY_NODES:
        return ALL_PATHWAY_NODES[cls_name][0], ALL_PATHWAY_NODES[cls_name][1]
    custom = get_custom_nodes()
    if cls_name in custom:
        return custom[cls_name]
    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════
# Edge computation
# ═══════════════════════════════════════════════════════════════════════

def compute_current_edges(active_set: Set[str]) -> List[Tuple[str, str]]:
    """Compute the effective edge list for the current active classes."""
    removed_set: Set[Tuple[str, str]] = set()
    for a, b in get_removed_edges():
        removed_set.add((a, b))
        removed_set.add((b, a))

    edges: List[Tuple[str, str]] = []
    for a, b in ALL_PATHWAY_EDGES:
        if a in active_set and b in active_set:
            if (a, b) not in removed_set and (b, a) not in removed_set:
                edges.append((a, b))
    for a, b in get_added_edges():
        if a in active_set and b in active_set:
            edges.append((a, b))
    return edges


# ═══════════════════════════════════════════════════════════════════════
# Layout reset
# ═══════════════════════════════════════════════════════════════════════

def reset_layout_state(active_classes: List[str]) -> None:
    """Reset all pathway layout state to a preset."""
    st.session_state[KEY_ACTIVE] = active_classes
    st.session_state[KEY_CUSTOM_NODES] = {}
    st.session_state[KEY_ADDED_EDGES] = []
    st.session_state[KEY_REMOVED_EDGES] = []
    st.session_state[KEY_POS_OVERRIDES] = {}
    for key in ['pathway_edge_to_remove_select',
                'pathway_add_edge_source',
                'pathway_add_edge_target',
                'pathway_move_node_select',
                'pathway_move_node_x',
                'pathway_move_node_y']:
        st.session_state.pop(key, None)