"""Truss 2D Problem."""

from typing import Any

from gymnasium import spaces
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.truss2d.model import utils
from engibench.problems.truss2d.model.conditions import Conditions
from engibench.problems.truss2d.model.config import Config
from engibench.problems.truss2d.model.constraints import calculate_overlap_score
from engibench.problems.truss2d.model.stiffness import calculate_stiffness
from engibench.problems.truss2d.model.visualization import viz
from engibench.problems.truss2d.model.volume_fraction import calculate_volume
from engibench.problems.truss2d.optimize.taea import Truss2dCTAEA


class Truss2D(Problem[npt.NDArray]):
    r"""Truss 2D integer optimization problem.

    ## Problem Description
    This is 2D topology optimization problem for maximizing the stiffness of a truss structure while minimizing material volume.

    ## Motivation
    Truss structures are fundamental components in civil, mechanical, and aerospace engineering, valued for their high strength-to-weight ratios.
    In real-world applications (e.g., bridge spans, crane arms, satellite chassis) engineers must balance the conflicting goals of structural integrity and material economy.
    This problem provides a sandbox for Topology Optimization (TO), where the goal is to discover the most efficient arrangement of members within a defined space.


    ## Design Space
    This multiobjective optimization problem is governed by linear elasticity.
    The problem is defined over a 2D domain, where nodal positions encode: fixed points, loaded points, and free points.
    A design is represented as a binary array, where each element indicates the presence (1) or absence (0) of a truss member between nodes.


    ## Objectives
    The optimization problem navigates the Pareto frontier between two primary
    competing goals:

    1. **Maximize Stiffness ($S$):**
       Measures the resistance to deformation. It is calculated as the ratio of
       total applied force to the total displacement at loaded nodes:
       $S = \sum \|F_{loaded}\| / \sum \|u_{loaded}\|$

    2. **Minimize Volume Fraction ($V_f$):**
       Reduces material cost and weight. Volume is the sum of the products of
       member lengths and their cross-sectional areas ($A = \pi r^2$):
       $V = \sum (A \times L_i)$


    ## Secondary Metrics (Monitored)
    - **Compliance:** Strain energy ($0.5 \cdot F^T u$).
    - **Member Stress:** Axial stress calculated via Hooke's Law ($\sigma = E \cdot \epsilon$).
    - **Euler Buckling:** Critical stress limit for members in compression:
      $\sigma_{cr} = (\pi^2 E r^2) / (4 L^2)$


    ## Constraints
    The primary constraint is to avoid overlapping truss members, which can make manufacturing infeasible.
    Other constraints calculated but not included in the optimization are: member normal stress, buckling stress, and deflection limits.


    ## Conditions
    Problem conditions are defined by creating a python dictionary with the following info:
    - 'nodes': A list of tuples representing the (x, y) coordinates of each node in the domain.
    - 'nodes_dof': A list of tuples representing the fixed (0) or free (1) degrees of freedom for each node.
    - 'load_conds': A list of lists of tuples representing the load conditions applied to specific nodes. Multiple load cases can be defined.
    - 'member_radii': A float encoding the radius of each truss member (fixed for all members).
    - 'young_modulus': A float encoding the Young's modulus of the truss material (Pascales).

    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("stiffness", ObjectiveDirection.MAXIMIZE),
        ("volume_fraction", ObjectiveDirection.MINIMIZE),
    )

    conditions = Conditions()
    design_space = spaces.MultiBinary(n=utils.get_num_bits(conditions))
    dataset_id = "IDEALLab/truss_2d_v0"
    container_id = None
    config: Config

    def __init__(self, seed: int = 0) -> None:
        """Initializes the truss2d problem.

        Args:
            seed (int): The random seed for the problem.
        """
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
        """
        super().reset(seed)

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> Any:
        """Simulates the performance of a truss design.

        Args:
            design (np.ndarray): The design to simulate encoded as a binary array.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.

        Returns:
            results (dict): A dictionary containing objective values and constraint values.
        """
        if config:
            self.conditions.update_from_dict(config)

        # 1. Calculate constraints
        constraint_results = calculate_overlap_score(self.conditions, design)
        constraint_score = constraint_results["score"]

        # 2. Calculate objectives
        volume = calculate_volume(self.conditions, design)
        stiffness, compliance, stress, buckling, deflection = calculate_stiffness(self.conditions, design)

        # 3. Return results
        return {
            "stiffness_avg": np.mean(stiffness),
            "stiffness_all": stiffness,
            "volume": volume,
            "member_overlaps": constraint_score,
            "compliance_avg": np.mean(compliance),
            "compliance_all": compliance,
            "stress": stress,
            "buckling": buckling,
            "deflection": deflection,
        }

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        """Tries to find the Pareto front for the current problem. The starting point should contain a set of initial designs, but can be left empty.

        Args:
            starting_point (npt.NDArray): The initial design population to optimize from.
            config (dict): A dictionary with the problem configuration optimization.

        Returns:
            Tuple[np.ndarray, list[OptiStep]]: The Pareto designs and their performance.
        """
        if config:
            self.conditions.update_from_dict(config)

        algorithm = Truss2dCTAEA(self, initial_designs=starting_point, population_size=100, generations=1000, node_sort_init=False)

        results = algorithm.solve()

        pareto_designs = results["X"]
        stiffness_values = results["Stiffness"]
        volume_values = results["Volume"]

        # Collect stiffness and volume pairs for each design and store in a unique opti_steps list
        opti_steps = []
        for i in range(pareto_designs.shape[0]):
            obj_values = np.array([stiffness_values[i], volume_values[i]], dtype=float)
            opti_steps.append(OptiStep(obj_values=obj_values, step=1))

        return pareto_designs, opti_steps

    def render(self, design: npt.NDArray, config: dict[str, Any] | None = None, *, open_window: bool = False) -> Figure:
        """Renders the truss design.

        Args:
            design (npt.NDArray): The design to render.
            config (dict): A dictionary with the problem configuration for the rendering.
            open_window (bool): Whether to open a window with the plot.

        Returns:
            Figure: The matplotlib figure containing the plot.
        """
        if config:
            self.conditions.update_from_dict(config)

        return viz(self, self.conditions, design, open_window=open_window)

    def random_design(self, dataset_split: str = "train", design_key: str = "optimal_design") -> tuple[npt.NDArray, int]:
        """Samples a valid random design from the dataset.

        Args:
            dataset_split (str): The key for the dataset to sample from.
            design_key (str): The key for the design to sample from.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = self.np_random.integers(low=0, high=10)
        if dataset_split == ""  or design_key == "":
            design = np.ones(shape=(self.design_space.n,), dtype=int)
        else:
            design = np.zeros(shape=(self.design_space.n,), dtype=int)

        return design, rnd


if __name__ == "__main__":
    # --- Create a new problem
    problem = Truss2D(seed=0)

    from engibench.problems.truss2d.case_studies.jmd_truss2d import case_study1

    starting_point = np.array([])
    pareto_designs, opti_steps = problem.optimize(starting_point, config=case_study1)

    # Visualization
    for pd in pareto_designs.tolist():
        problem.render(pd, open_window=True)
