"""Thermo Elastic 3D Problem."""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any, ClassVar

from gymnasium import spaces
import napari
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.thermoelastic3d.model.fem_model import FeaModel3D

NELX = NELY = NELZ = 16
FIXED_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FIXED_ELEMENTS[0, 0, 0] = 1
FIXED_ELEMENTS[0, -1, -1] = 1
FIXED_ELEMENTS[0, -1, 0] = 1
FORCE_ELEMENTS_X = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_X[-1, -1, -1] = 1
FORCE_ELEMENTS_Y = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_Y[-1, -1, -1] = 1
FORCE_ELEMENTS_Z = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_Z[-1, -1, -1] = 1
HEATSINK_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
HEATSINK_ELEMENTS[-1, -1, 0] = 1


class ThermoElastic3D(Problem[npt.NDArray]):
    r"""Truss 3D integer optimization problem.

    ## Problem Description
    This is 3D topology optimization problem for minimizing weakly coupled thermo-elastic compliance subject to boundary conditions and a volume fraction constraint.

    ## Motivation
    As articulated in their respective sections, both the Beams2D and HeatConduction2D problems found in the EngiBench library are fundamental engineering design problems that have historically served as benchmarks for the development and testing of optimization methods.
    While their relevance is supported by needs in real engineering design scenarios (aerospace, automotive, consumer electronics, etc...), their mono-domain nature ignores the reality that coupling between domains exists, and should be accounted for in scenarios where performance in one domain significantly impacts performance in another.
    To address this distinction, a multi-physics topology optimization problem is developed that captures the coupling between structural and thermal domains in three dimensions.

    ## Design space
    This multi-physics topology optimization problem is governed by linear elasticity and steady-state heat conduction with a one-way coupling from the thermal domain to the elastic domain.
    The problem is defined over a cube 3D domain, where load elements and support elements are placed along the boundary to define a unique elastic condition.
    Similarly, heatsink elements are placed along the boundary to define a unique thermal condition.
    The design space is then defined by a 3D array representing density values (parameterized by DesignSpace = [0,1]^{nelx x nely x nelz}, where nelx, nely, and nelz denote the x, y, and z dimensions respectively).

    ## Objectives
    The objective of this problem is to minimize total compliance C under a volume fraction constraint V by placing a thermally conductive material.
    Total compliance is defined as the sum of thermal compliance and structural compliance.

    ## Conditions
    Problem conditions are defined by creating a python dict with the following info:
    - `fixed_elements`: Encodes a binary NxNxN matrix of the structurally fixed elements in the domain.
    - `force_elements_x`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the x-direction.
    - `force_elements_y`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the y-direction.
    - `force_elements_z`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the z-direction.
    - `heatsink_elements`: Encodes a binary NxNxN matrix specifying elements that have a heat sink.
    - `volfrac`: Encodes the target volume fraction for the volume fraction constraint.
    - `rmin`: Encodes the filter size used in the optimization routine.
    - `weight`: Allows one to control which objective is optimized for. 1.0 Is pure structural optimization, while 0.0 is pure thermal optimization.

    ## Simulator
    The simulation code is based on a Python adaptation of the popular 88-line topology optimization code, modified to handle the thermal domain in addition to thermal-elastic coupling.
    Optimization is conducted by reformulating the integer optimization problem as a continuous one (leveraging a SIMP approach), where a density filtering approach is used to prevent checkerboard-like artifacts.
    The optimization process itself operates by calculating the sensitivities of the design variables with respect to total compliance (done efficiently using the Adjoint method), calculating the sensitivities of the design variables with respect to the constraint value, and then updating the design variables by solving a convex-linear subproblem and taking a small step (using the method of moving asymptotes).
    The optimization loop terminates when either an upper bound of the number of iterations has been reached or if the magnitude of the gradient update is below some threshold.

    ## Dataset
    The dataset linked to this problem is on huggingface [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/thermoelastic_3d_v0).
    This dataset contains a set of 100 optimized thermoelastic designs in a 16x16x16 domain, where each design is optimized for a unique set of conditions.
    Each datapoint's conditions are randomly generated by arbitrarily placing: a single loaded element along the bottom boundary, two fixed elements (fixed in the x, y, and z direction) along the left and top boundary, and heatsink elements along the right boundary.
    Furthermore, values for the volume fraction constraint are randomly selected in the range $[0.2, 0.5]$.

    Relevant datapoint fields include:
    - `optimal_design`: An optimized design for the set of boundary conditions
    - `fixed_elements`: Encodes a binary NxNxN matrix of the structurally fixed elements in the domain.
    - `force_elements_x`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the x-direction.
    - `force_elements_y`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the y-direction.
    - `force_elements_z`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the z-direction.
    - `heatsink_elements`: Encodes a binary NxNxN matrix specifying elements that have a heat sink.
    - `volume_fraction`: The volume fraction value of the optimized design
    - `structural_compliance`: The structural compliance of the optimized design
    - `thermal_compliance`: The thermal compliance of the optimized design
    - `nelx`: The number of elements in the x-direction
    - `nely`: The number of elements in the y-direction
    - `nelz`: The number of elements in the z-direction
    - `volfrac`: The volume fraction target of the optimized design
    - `rmin`: The filter size used in the optimization routine
    - `weight`: The domain weighting used in the optimization routine

    ## Lead
    Gabriel Apaza @gapaza
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("structural_compliance", ObjectiveDirection.MINIMIZE),
        ("thermal_compliance", ObjectiveDirection.MINIMIZE),
        ("volume_fraction", ObjectiveDirection.MINIMIZE),
    )

    @dataclass
    class Conditions:
        """Conditions."""

        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FIXED_ELEMENTS
        )
        force_elements_x: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_X
        )
        force_elements_y: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Y
        )
        force_elements_z: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Z
        )
        heatsink_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: HEATSINK_ELEMENTS
        )
        volfrac: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.3
        rmin: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=3.0).warning().category(IMPL)
        ] = 1.5
        weight: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """1.0 for pure structural, 0.0 for pure thermal"""

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(NELX, NELY, NELZ), dtype=np.float32)
    dataset_id = "IDEALLab/thermoelastic_3d_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        nelx: ClassVar[Annotated[int, bounded(lower=1).category(THEORY)]] = NELX
        nely: ClassVar[Annotated[int, bounded(lower=1).category(THEORY)]] = NELY
        nelz: ClassVar[Annotated[int, bounded(lower=1).category(THEORY)]] = NELZ

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int, nelz: int) -> None:
            """Constraint for rmin ∈ (0.0, max{ nelx, nely, nelz }]."""
            assert 0.0 < rmin <= max(nelx, nely, nelz), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely, nelz)]"

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> npt.NDArray:
        """Simulates the performance of a design topology.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        boundary_dict = dataclasses.asdict(self.conditions)
        for key, value in (config or {}).items():
            if key in boundary_dict:
                if isinstance(value, list):
                    boundary_dict[key] = np.array(value)
                else:
                    boundary_dict[key] = value

        results = FeaModel3D(plot=False, eval_only=True).run(boundary_dict, x_init=design)
        return np.array([results["structural_compliance"], results["thermal_compliance"], results["volume_fraction"]])

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes a topology for the current problem. Note that an appropriate starting_point for the optimization is defined by a uniform material distribution equal to the volume fraction constraint.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        boundary_dict = dataclasses.asdict(self.conditions)
        boundary_dict.update({k: v for k, v in (config or {}).items() if k in boundary_dict})
        results = FeaModel3D(plot=False, eval_only=False).run(boundary_dict, x_init=starting_point)
        design = np.array(results["design"]).astype(np.float32)
        opti_steps = results["opti_steps"]
        return design, opti_steps

    def render(self, design: np.ndarray, *, open_window: bool = False) -> np.ndarray:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            fig (np.ndarray): The rendered design.
        """
        design = np.array(design)
        design = np.transpose(design, (2, 0, 1))

        viewer = napari.Viewer()
        viewer.add_image(design, name="rho", rendering="attenuated_mip")
        viewer.dims.ndisplay = 3  # switch to 3D view
        if open_window is True:
            napari.run()
        return viewer.export_figure(flash=False)

    def random_design(self, dataset_split: str = "train", design_key: str = "optimal_design") -> tuple[npt.NDArray, int]:
        """Samples a valid random design.

        Args:
            dataset_split (str): The key for the dataset to sample from.
            design_key (str): The key for the design to sample from.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split]), dtype=int)
        return np.array(self.dataset[dataset_split][design_key][rnd]), rnd


if __name__ == "__main__":
    # --- Create a new problem
    problem = ThermoElastic3D(seed=0)

    # --- Load the problem dataset
    dataset = problem.dataset
    first_item = dataset["train"][0]
    first_item_design = np.array(first_item["optimal_design"])
    problem.render(first_item_design, open_window=True)

    # --- Render the design
    design, _ = problem.random_design()
    problem.render(design, open_window=True)

    # --- Optimize a design ---
    design = 0.2 * np.ones((NELX, NELY, NELZ), dtype=float)
    design, objectives = problem.optimize(design)
    problem.render(design, open_window=True)

    # --- Evaluate a design ---
    problem.reset(seed=0)
    design, _ = problem.random_design()
    print(problem.simulate(design))
