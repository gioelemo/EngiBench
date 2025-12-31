"""Helper methods for visualizing a 2D truss design."""

from __future__ import annotations

import textwrap
from typing import Any, TYPE_CHECKING

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from engibench.problems.truss2d.model import utils

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from engibench.problems.truss2d.model.conditions import Conditions
    from engibench.problems.truss2d.v0 import Truss2D

PLOT_GRID = True


def viz( # noqa: PLR0915 PLR0912 C901
        model: Truss2D, conditions: Conditions, design_rep: Any, *, open_window: bool = False) -> Figure:
    """Visualizes a 2D truss design.

    Args:
        model (Truss2D): The truss problem model.
        conditions: The conditions object containing problem definition.
        design_rep: The design representation to visualize.
        open_window (bool): Whether to open a window with the plot.

    Returns:
        Figure: The matplotlib figure containing the plot.
    """
    # If design is a numpy array, convert to list
    if isinstance(design_rep, np.ndarray):
        design_rep = design_rep.tolist()

    n_loads = len(conditions.load_conds)
    load_conds = conditions.load_conds

    # Calculate number of rows and columns for subplots
    n_loads_extra = n_loads + 1
    cols = int(np.ceil(np.sqrt(n_loads_extra)))
    rows = int(np.ceil(n_loads_extra / cols))

    # Create GridSpec layout
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(3.5 * cols, 3.5 * rows), dpi=150)  # Adjust the figure size based on rows and cols


    for l_idx, load_cond in enumerate([None, *load_conds]):
        if l_idx == 0:
            continue

        row = l_idx // cols
        col = l_idx % cols
        plt.subplot(gs[row, col])

        _, _, node_idx_pairs, _ = utils.convert(conditions, design_rep)
        nodal_locations = conditions.nodes

        # Plotting the truss members
        for start, end in node_idx_pairs:
            x_coords = [nodal_locations[start][0], nodal_locations[end][0]]
            y_coords = [nodal_locations[start][1], nodal_locations[end][1]]
            plt.plot(x_coords, y_coords, "black")  # Plot truss members as red lines

        # Get axis
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Plotting the nodes
        for i, (x, y) in enumerate(nodal_locations):
            node_dof = conditions.nodes_dof[i]
            if node_dof[0] == 0 or node_dof[1] == 0:  # Node restrained in at least 1 dof
                plt.plot(x, y, "ro")  # Plot nodes as red squares
                if node_dof[0] == 0:
                    arrow_start = [x, y]
                    arrow_end = [x + (x_range / 10), y]
                    ax.arrow(
                        arrow_start[0],
                        arrow_start[1],
                        arrow_end[0] - arrow_start[0],
                        arrow_end[1] - arrow_start[1],
                        head_width=0.2,
                        head_length=0.2,
                        fc="red",
                        ec="red",
                        width=0.05,
                        zorder=2,
                    )
                if node_dof[1] == 0:
                    arrow_start = [x, y]
                    arrow_end = [x, y + (y_range / 10)]
                    ax.arrow(
                        arrow_start[0],
                        arrow_start[1],
                        arrow_end[0] - arrow_start[0],
                        arrow_end[1] - arrow_start[1],
                        head_width=0.2,
                        head_length=0.2,
                        fc="red",
                        ec="red",
                        width=0.05,
                        zorder=2,
                    )
            else:
                plt.plot(x, y, "bo")  # Plot nodes as blue circles

            plt.text(x, y, f"{i}", fontsize=12, ha="right")  # Annotate nodes with their index

        # Node Loads
        nodes_loads = load_cond
        ax = plt.gca()  # Get the current axis
        for i, (load_x, load_y) in enumerate(nodes_loads):
            if load_x != 0:
                arrow_start = [nodal_locations[i][0], nodal_locations[i][1]]
                if load_x > 0:
                    arrow_end = [arrow_start[0] + (x_range / 10), arrow_start[1]]
                else:
                    arrow_end = [arrow_start[0] - (x_range / 10), arrow_start[1]]
                ax.arrow(
                    arrow_start[0],
                    arrow_start[1],
                    arrow_end[0] - arrow_start[0],
                    arrow_end[1] - arrow_start[1],
                    head_width=0.2,
                    head_length=0.2,
                    width=0.05,
                    fc="green",
                    ec="green",
                    zorder=2,
                )
                plt.text(
                    arrow_end[0], arrow_end[1] + (y_range / 100), f"{load_x:.2f} N", fontsize=10, ha="right"
                )  # Annotate nodes with their index

            if load_y != 0:
                arrow_start = [nodal_locations[i][0], nodal_locations[i][1]]
                if load_y > 0:
                    arrow_end = [arrow_start[0], arrow_start[1] + (y_range / 10)]
                else:
                    arrow_end = [arrow_start[0], arrow_start[1] - (y_range / 10)]
                ax.arrow(
                    arrow_start[0],
                    arrow_start[1],
                    arrow_end[0] - arrow_start[0],
                    arrow_end[1] - arrow_start[1],
                    head_width=0.2,
                    head_length=0.2,
                    width=0.05,
                    fc="green",
                    ec="green",
                    zorder=2,
                )
                plt.text(
                    arrow_end[0], arrow_end[1] - (x_range / 100), f"{load_y:.2f} N", fontsize=10, ha="right"
                )  # Annotate nodes with their index

        # Setting the plot labels and title
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"Load Condition {l_idx}")
        plt.grid(PLOT_GRID)
        plt.axis("equal")  # Ensure the aspect ratio is equal to avoid distortion

    # Design Text
    design_metrics = get_design_metrics(model, conditions, design_rep)
    wrapped_lines = []
    for paragraph in design_metrics.split("\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph, 50))
    wrapped_text = "\n".join(wrapped_lines)  # Remove the last empty line
    plt.figtext(
        0.05, 0.5,
        wrapped_text,
        ha="left", va="bottom", fontsize=9,
        bbox={"facecolor":"grey", "alpha":0.5},
    )
    plt.tight_layout()

    if open_window is True:
        plt.show()
    plt.close("all")
    return fig





def get_design_metrics(model: Truss2D, conditions: Conditions, design_rep: Any) -> str:
    """Determines text for visualization of design metrics.

    Args:
        model (Truss2D): The truss problem model.
        conditions (Conditions): The problem conditions.
        design_rep: The design representation to evaluate.

    Returns:
        metrics_text (str): The design metrics text.
    """
    bit_list, _, _, _ = utils.convert(conditions, design_rep)
    num_members = sum(bit_list)
    used_nodes = utils.get_used_nodes(conditions, design_rep)
    results = model.simulate(design_rep)
    vol_frac = results["volume"]
    stiff = results["stiffness_all"]

    metrics = [
        f"Truss Members: {num_members}",
        f"Truss Nodes: {used_nodes}",
        "-----",
        f"Youngs Modulus: {conditions.young_modulus:.2e} Pa (N/m^2)",
        f"Member Radii: {conditions.member_radii} m",
        f"Volume Fraction: {vol_frac:.3f}",
    ]

    stiffness_metrics = []
    for idx, s in enumerate(stiff):
        load_cond_metrics = ["-----"]
        load_cond_metrics.append(f"Load Condition {idx + 1}")
        load_cond_metrics.append(f"Stiffness: {s:.2e} N/m")
        stiffness_metrics.extend(load_cond_metrics)
    metrics.extend(stiffness_metrics)

    return "\n".join(metrics)
