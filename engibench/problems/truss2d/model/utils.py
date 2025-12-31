"""Utility functions for 2D truss modeling in EngiBench."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # This import ONLY happens during type checking
    from engibench.problems.truss2d.model.conditions import Conditions


def convert( # noqa: PLR0912 C901
    conditions: Conditions, orig_rep: Any
) -> tuple[list[int], str, list[tuple[float, float]], list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Converts a design representation to four possible representations.

    Args:
        conditions (Conditions): The problem conditions containing node information.
        orig_rep (Any): The original design representation (e.g., binary array).

    Returns:
        bit_list (list[int]): Design in bit list representation.
        bit_str (str): Design in bit string representation.
        node_idx_pairs (list[tuple[float, float]]): List of node index pairs.
        node_coords (list[tuple[tuple[float, float], tuple[float, float]]]): List of node coordinate pairs.
    """
    nodes = conditions.nodes
    bit_members = get_bit_members(conditions)

    bit_list = None
    bit_str = ""
    node_idx_pairs = None
    node_coords = None

    # Depending on representation, convert to bit list
    if isinstance(orig_rep, str):
        bit_str = orig_rep
        bit_list = []
        for char in bit_str:
            bit_list.append(int(char))
    elif isinstance(orig_rep, list):
        first_element = orig_rep[0]
        if isinstance(first_element, int):
            bit_list = orig_rep
        elif isinstance(first_element, (tuple, list)):
            ff_element = first_element[0]
            if isinstance(ff_element, int):  # Node index pairs
                node_idx_pairs = orig_rep
                bit_list = []
                for bm in bit_members:
                    if contains_pair(bm, node_idx_pairs):
                        bit_list.append(1)
                    else:
                        bit_list.append(0)
            else:  # Node coordinate pairs
                node_coords = orig_rep
                # Convert to node index pairs
                node_idx_pairs = []
                for node_pair in node_coords:
                    idx_pair = [gcoords_to_node_idx(coord_pair, nodes) for coord_pair in node_pair]
                    node_idx_pairs.append(idx_pair)
                # Convert to bit list
                bit_list = []
                for bm in bit_members:
                    if contains_pair(bm, node_idx_pairs):
                        bit_list.append(1)
                    else:
                        bit_list.append(0)

    if bit_str == "" and bit_list is not None:
        bit_str = "".join([str(bit) for bit in bit_list])
    if node_idx_pairs is None:
        node_idx_pairs = []
        for idx, bit in enumerate(bit_list):
            if bit == 1:
                node_idx_pairs.append(bit_members[idx])
    if node_coords is None:
        node_coords = []
        for pair in node_idx_pairs:
            node_coords.append([nodes[pair[0]], nodes[pair[1]]])

    return bit_list, bit_str, node_idx_pairs, node_coords


def get_num_bits(conditions: Conditions):
    """Returns the number of possible truss members for a given problem condition.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        num_bits (int): The number of possible truss members.
    """
    num_nodes = len(conditions.nodes)
    return int(num_nodes * (num_nodes - 1) / 2)


def sort_nodes(conditions: Conditions) -> list[tuple[float, float]]:
    """Sorts the nodes in the conditions based on their coordinates.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        sorted_nodes (list[tuple[float, float]]): The sorted list of node coordinates
    """
    return conditions.apply_sorting().nodes


def get_bit_members(conditions: Conditions) -> list[tuple[int, int]]:
    """Generates a list identifying which bits correspond to which members (node index pairs).

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        bit_members (list[tuple[int, int]]): List of node index pairs corresponding to each bit.
    """
    nodes = conditions.nodes
    bit_members = []
    for idx, _node in enumerate(nodes):
        for idx2, _node2 in enumerate(nodes):
            if idx2 > idx:
                bit_members.append((idx, idx2))
    return bit_members


def contains_pair(pair: tuple[int, int], pairs: list[tuple[int, int]]) -> bool:
    """Checks if a given node index pair is contained in a list of node index pairs.

    Args:
        pair (tuple[int, int]): The node index pair to check.
        pairs (list[tuple[int, int]]): The list of node index pairs.

    Returns:
        bool: True if the pair is contained in the list, False otherwise.
    """
    return any(equate_pairs(pair, p) is True for p in pairs)


def equate_pairs(node1: tuple[int, int], node2: tuple[int, int]) -> bool:
    """Checks if two node index pairs are equivalent (if they represent the same member).

    Args:
        node1 (tuple[int, int]): First node index pair.
        node2 (tuple[int, int]): Second node index pair.

    Returns:
        bool: True if the pairs are equivalent, False otherwise.
    """
    return bool(node1[0] in node2 and node1[1] in node2)


def gcoords_to_node_idx(coords: tuple[float, float], nodes: list[tuple[float, float]]) -> int:
    """Converts global coordinates to a node index.

    Args:
        coords (tuple[float, float]): The global coordinates of the node.
        nodes (list[tuple[float, float]]): The list of node coordinates.

    Returns:
        idx (int): The index of the node corresponding to the given coordinates.
    """
    for idx, node in enumerate(nodes):
        if coords[0] == node[0] and coords[1] == node[1]:
            return idx
    raise ValueError("Node not found in nodes")


def get_bit_coords(conditions: Conditions):
    """Generates the node coordinate pairs associated with each bit.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        bit_coords (list[tuple[tuple[float, float], tuple[float, float]]]): List of node coordinate pairs corresponding to each bit.
    """
    nodes = conditions.nodes
    bit_coords = []
    for idx, node in enumerate(nodes):
        for idx2, node2 in enumerate(nodes):
            if idx2 > idx:
                bit_coords.append((node, node2))
    return bit_coords


def get_load_nodes(load_conds: list[tuple[float, float]]) -> list[int]:
    """Generates a list of node indices that have loads applied.

    Args:
        load_conds (list[tuple[float, float]]): The list of load conditions for each node.

    Returns:
        load_nodes (list[int]): List of node indices with applied loads.
    """
    load_nodes = []
    for idx, load in enumerate(load_conds):
        if load[0] != 0.0 or load[1] != 0.0:
            load_nodes.append(idx)
    return load_nodes


def get_edge_nodes(conditions: Conditions) -> list[int]:
    """Generates a list of node indices that are on the edges of the domain.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        edge_nodes (list[int]): List of node indices on the edges of the domain.
    """
    nodes = conditions.nodes
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    edge_nodes = []
    for idx, node in enumerate(nodes):
        if node[0] == x_min or node[0] == x_max or node[1] == y_min or node[1] == y_max:
            edge_nodes.append(idx)
    return edge_nodes


def get_all_fixed_nodes(conditions: Conditions) -> list[int]:
    """Generates a list of node indices that are fixed in any degrees of freedom.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        fixed_nodes (list[int]): List of node indices that are fixed.
    """
    nodes_dof = conditions.nodes_dof
    fixed_nodes = set()
    for idx, dof in enumerate(nodes_dof):
        if 0 in dof:
            fixed_nodes.add(idx)
    return list(fixed_nodes)


def get_fully_fixed_nodes(conditions: Conditions) -> list[int]:
    """Generates a list of node indices that are fully fixed in all degrees of freedom.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        fully_fixed_nodes (list[int]): List of node indices that are fully fixed.
    """
    nodes_dof = conditions.nodes_dof
    fully_fixed_nodes = []
    for idx, dof in enumerate(nodes_dof):
        if dof[0] == 0 and dof[1] == 0:
            fully_fixed_nodes.append(idx)
    return fully_fixed_nodes


def get_free_nodes(conditions: Conditions) -> list[int]:
    """Generates a list of node indices that are free in all degrees of freedom.

    Args:
        conditions (Conditions): The problem conditions containing node information.

    Returns:
        free_nodes (list[int]): List of node indices that are free.
    """
    nodes_dof = conditions.nodes_dof
    free_nodes = set()
    for idx, dof in enumerate(nodes_dof):
        if 0 not in dof:
            free_nodes.add(idx)
    return list(free_nodes)


def get_used_nodes(conditions: Conditions, design: Any) -> list[int]:
    """Generates a list of node indices that are used in the current design.

    Args:
        conditions (Conditions): The problem conditions containing node information.
        design (Any): The design representation (e.g., binary array).

    Returns:
        used_nodes (list[int]): List of node indices used in the design.
    """
    _, _, node_idx_pairs, _ = convert(conditions, design)
    used_nodes = set()
    for pair in node_idx_pairs:
        for node in pair:
            used_nodes.add(node)
    return list(used_nodes)


def get_design_text(conditions: Conditions, design: Any) -> str:
    """Generates a human-readable text representation of the design.

    Args:
        conditions (Conditions): The problem conditions containing node information.
        design (Any): The design representation (e.g., binary array).

    Returns:
        design_text (str): Human-readable text representation of the design.
    """
    bit_list, _, _, _ = convert(conditions, design)
    str_members = [str(x) for x in bit_list]
    design_text = "".join(str_members)
    return "[" + design_text + "]"


def get_node_connections(conditions: Conditions, design: Any, node_idx: int) -> list[tuple[int, int]]:
    """Generates a list of node index pairs that are connected to a given node in the design.

    Args:
        conditions (Conditions): The problem conditions containing node information.
        design (Any): The design representation (e.g., binary array).
        node_idx (int): The index of the node to check connections for.

    Returns:
        connected_pairs (list[tuple[int, int]]): List of node index pairs connected to the given node.
    """
    _, _, node_idx_pairs, _ = convert(conditions, design)
    return [pair for pair in node_idx_pairs if node_idx in pair]
