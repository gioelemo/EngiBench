"""Methods for calculating the volume fraction of a 2D truss design."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np

from engibench.problems.truss2d.model import utils

if TYPE_CHECKING:
    from engibench.problems.truss2d.model.conditions import Conditions


def calculate_volume(conditions: Conditions, design_rep: Any) -> float:
    """Sums the volume of all the truss members in a design (not accounting for overlaps at nodes).

    Args:
        conditions: The conditions object containing problem definition.
        design_rep: The design representation to evaluate.

    Returns:
        volume: The total volume of the truss design.
    """
    _, _, node_idx_pairs, _ = utils.convert(conditions, design_rep)

    nodes = deepcopy(conditions.nodes)
    nodes = np.array(nodes).tolist()
    member_radii = conditions.member_radii
    radius = member_radii
    member_cs_area = (np.pi * radius) ** 2
    member_vols = []
    for ca in node_idx_pairs:
        p1 = nodes[ca[0]]
        p2 = nodes[ca[1]]
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        member_vol = dist * member_cs_area
        member_vols.append(member_vol)
    return sum(member_vols)
