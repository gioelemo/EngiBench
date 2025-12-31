"""Sets up the configuration for the 2D truss problem."""

from dataclasses import dataclass

from engibench.problems.truss2d.model.conditions import Conditions


@dataclass
class Config(Conditions):
    """Configuration for the 2D Truss Problem."""
