"""Utility functions for calculating the constraints for a 2D truss problem."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from engibench.problems.truss2d.model import utils

if TYPE_CHECKING:
    from engibench.problems.truss2d.model.conditions import Conditions


class Point:
    """A simple class to represent a 2D point."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_overlap_score(conditions: Conditions, design_rep: Any) -> dict[str, Any]:
    """Calculates how many truss members overlap or cross each other.

    Args:
        conditions: The conditions object containing problem definition.
        design_rep: The design representation to evaluate.

    Returns:
        A dictionary with the total score, number of crossings, and number of collinear overlaps.
    """
    # 1. Standardize the input
    # (Assuming convert is available in the scope or imported)
    _, _, node_idx_pairs, node_coords_pairs = utils.convert(conditions, design_rep)

    crossing_violations = 0
    collinear_violations = 0

    num_members = len(node_idx_pairs)

    # 2. Iterate through every unique pair of members
    for i in range(num_members):
        for j in range(i + 1, num_members):
            # Indices
            indices_1 = node_idx_pairs[i]
            indices_2 = node_idx_pairs[j]

            # Check for shared node
            set_1 = set(indices_1)
            set_2 = set(indices_2)
            shared = set_1.intersection(set_2)
            has_shared_node = len(shared) > 0

            # Coordinates
            coords_1 = node_coords_pairs[i]  # [[x1, y1], [x2, y2]]
            coords_2 = node_coords_pairs[j]  # [[x3, y3], [x4, y4]]

            p1 = Point(*coords_1[0])
            q1 = Point(*coords_1[1])
            p2 = Point(*coords_2[0])
            q2 = Point(*coords_2[1])

            # 3. Perform Geometry Check
            overlap_type = check_overlap_type(p1, q1, p2, q2)

            if overlap_type == "collinear":
                if not has_shared_node:
                    # Case A: Disjoint but overlapping (e.g. parallel lines merging)
                    # Always a violation
                    collinear_violations += 1
                else:
                    # Case B: Collinear AND Share a node.
                    # We must distinguish between:
                    # 1. Valid: A-B-C (Opposite directions from B)
                    # 2. Invalid: A-B and A-C (Same direction from A, i.e., overlap)

                    # Identify the shared node coordinate and the two "tail" coordinates
                    shared_idx = next(iter(shared))

                    # Find coordinates of the shared node
                    # (We simply match the index to the coord list)
                    if indices_1[0] == shared_idx:
                        shared_pt = coords_1[0]
                        tail_1 = coords_1[1]
                    else:
                        shared_pt = coords_1[1]
                        tail_1 = coords_1[0]

                    tail_2 = coords_2[1] if indices_2[0] == shared_idx else coords_2[0]

                    # Create vectors from the shared node to the tails
                    vec1 = (tail_1[0] - shared_pt[0], tail_1[1] - shared_pt[1])
                    vec2 = (tail_2[0] - shared_pt[0], tail_2[1] - shared_pt[1])

                    # Calculate Dot Product
                    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

                    # If Dot Product > 0, vectors point in same direction -> OVERLAP
                    # If Dot Product < 0, vectors point in opposite directions -> NO OVERLAP (Straight line)
                    if dot_product > 0:
                        collinear_violations += 1

            elif overlap_type == "crossing" and not has_shared_node:
                # If they cross but share a node, it's just a joint (V-shape), which is valid.
                crossing_violations += 1

    total_score = crossing_violations + collinear_violations

    return {"score": total_score, "crossings": crossing_violations, "collinear": collinear_violations}


def check_overlap_type(p1: Point, q1: Point, p2: Point, q2: Point) -> str:
    """Determines the type of overlap between two line segments defined by points p1-q1 and p2-q2.

    Args:
        p1: Start point of the first line segment.
        q1: End point of the first line segment.
        p2: Start point of the second line segment.
        q2: End point of the second line segment.

    Returns:
        cross_type (str): "crossing", "collinear", or "none".
    """

    # 1. Orientation helper
    def orientation(p: Point, q: Point, r: Point) -> int:
        """Find the orientation of an ordered triplet (p, q, r)."""
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock wise

    # 2. On-segment helper
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        """Check if point q lies on segment pr."""
        return (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return "crossing"

    # Special Cases for collinearity
    if o1 == 0 and on_segment(p1, p2, q1):
        return "collinear"
    if o2 == 0 and on_segment(p1, q2, q1):
        return "collinear"
    if o3 == 0 and on_segment(p2, p1, q2):
        return "collinear"
    if o4 == 0 and on_segment(p2, q1, q2):
        return "collinear"

    return "none"
