"""Defines the Conditions for a 2D Truss Problem."""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Annotated

from engibench.constraint import bounded
from engibench.constraint import THEORY

NODES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
]
NODES_DOF = [  # Encodes which degrees of freedom are fixed for each node (0 = fixed, 1 = free)
    (0, 0),
    (1, 1),
    (1, 1),
    (1, 1),
    (0, 0),
    (1, 1),
    (1, 1),
    (1, 1),
    (0, 0),
    (1, 1),
    (1, 1),
    (1, 1),
    (0, 0),
    (1, 1),
    (1, 1),
    (1, 1),
]
LOAD_CONDS = [  # Encodes the loads applied to each node in each direction (newtons)
    [  # Multiple load conditions can be specified
        (0, 0),
        (0, 0),
        (0, 1),
        (-1, 1),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]
]
MEMBER_RADII = 0.1  # meters
YOUNGS_MODULUS = 1.8162e6  # Pascals (N/m^2)


@dataclass
class Conditions:
    """Conditions for a 2D Truss Problem."""

    nodes: Annotated[list[tuple[float, float]], bounded(lower=0.0, upper=1e10).category(THEORY)] = field(
        default_factory=lambda: NODES
    )
    nodes_dof: Annotated[list[tuple[int, int]], bounded(lower=0.0, upper=1e10).category(THEORY)] = field(
        default_factory=lambda: NODES_DOF
    )
    load_conds: Annotated[list[list[tuple[float, float]]], bounded(lower=0.0, upper=1e10).category(THEORY)] = field(
        default_factory=lambda: LOAD_CONDS
    )
    member_radii: Annotated[float, bounded(lower=0.0, upper=1e10).category(THEORY)] = MEMBER_RADII
    young_modulus: Annotated[float, bounded(lower=0.0, upper=1e10).category(THEORY)] = YOUNGS_MODULUS

    def __post_init__(self):
        """Sort automatically on initialization."""
        self.apply_sorting()

    def apply_sorting(self):
        """Re-synchronizes all fields based on the current order of 'nodes'.

        Returns:
            Conditions: The updated Conditions object with sorted fields.
        """
        if not self.nodes:
            return self

        # Determine the sorted order (lexicographical: x then y)
        sorted_indices = sorted(range(len(self.nodes)), key=lambda i: self.nodes[i])

        # Reorder everything based on those indices
        self.nodes = [self.nodes[i] for i in sorted_indices]
        self.nodes_dof = [self.nodes_dof[i] for i in sorted_indices]
        self.load_conds = [[case[i] for i in sorted_indices] for case in self.load_conds]

        return self  # Allows for chaining: obj.apply_sorting().some_other_method()

    def update_from_dict(self, data: dict):
        """Updates fields from a dict, ignoring keys not in the dataclass."""
        # Get a set of valid field names for this dataclass
        valid_fields = {f.name for f in fields(self)}

        for key, value in data.items():
            if key in valid_fields:
                setattr(self, key, value)

        # Re-sort because 'nodes' (or other dependent fields) might have changed
        self.apply_sorting()
