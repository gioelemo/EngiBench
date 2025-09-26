# ruff: noqa: N806
# Disabled variable name conventions

"""Beams 2D problem."""

from copy import deepcopy
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any

from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import Criticality
from engibench.constraint import greater_than
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import image_to_design
from engibench.problems.beams2d.backend import inner_opt
from engibench.problems.beams2d.backend import overhang_filter_d
from engibench.problems.beams2d.backend import overhang_filter_x
from engibench.problems.beams2d.backend import State
from engibench.utils.upcast import upcast


@dataclass
class ExtendedOptiStep(OptiStep):
    """Extended OptiStep to store a single NumPy array representing a density field at a given optimization step."""

    design: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))


@constraint(categories=THEORY, criticality=Criticality.Warning)
def volume_fraction_bound(design: npt.NDArray, volfrac: float) -> None:
    """Constraint for volume fraction of the design."""
    actual_volfrac = design.mean()
    tolerance = 0.01
    assert abs(actual_volfrac - volfrac) <= tolerance, (
        f"Volume fraction of the design {actual_volfrac:.4f} does not match target {volfrac:.4f} specified in the conditions. While the optimizer might fix it, this is likely to affect objective values as the initial design is not feasible given the constraints."
    )


class Beams2D(Problem[npt.NDArray]):
    r"""Beam 2D topology optimization problem.

    ## Problem Description
    Beams2D is a structural topology optimization (TO) problem that optimizes a 2D Messerschmitt-Bölkow-Blohm
    (MBB) beam under bending. The beam is symmetric about the central vertical axis, with a force applied
    at the top; only the right half is modeled in our case. Problems are formulated using density-based TO,
    drawing from an existing Python [implementation](https://github.com/arjendeetman/TopOpt-MMA-Python).

    ## Motivation
    The optimization of beam cross-sections is one of a fundamental problem in engineering, aiming to
    maximize the structural stiffness under some applied force. This objective is usually formulated as
    minimizing the compliance, which is the inverse of stiffness. In particular, TO frames the problem as
    one of optimal material distribution, defining a grid of elements for which the material densities must
    be determined on a scale from 0 to 1, where 1 represents the presence of material. After applying the
    beam loads and other boundary conditions, designs are typically optimized using a gradient-based
    approach with the help of the finite element method (FEM). While this is one of the simplest
    TO applications, it is still a computationally expensive process requiring many iterations, opening the
    door for faster approximation methods such as generative inverse design.
    One of the most common beam types in TO is the Messerschmitt-Bölkow-Blohm (MBB) beam,
    which is supported at the bottom-right and bottom-left corners, with a downward force applied
    on the top-center. Given this symmetric configuration, one half of the design may be optimized
    while representing the entire structure. We implement the MBB beam in ENGIBENCH for the most
    accessible comparison to previous works in this domain.

    ## Design Space
    This problem simulates the right half-section of a MBB beam under bending. This half-beam is
    subjected to a force at its top-left corner (corresponding to the top-center of the entire design) which
    may also be shifted to the right to simulate different loading conditions. A roller support at the
    bottom-right corner prevents vertical movement, and a symmetric boundary condition is enforced on
    the left edge. The design space is an array of solid densities in `[0., 1.]` with a default size of
    `(100, 50)` used by default, where `nelx = 100` and `nely = 50`. Internally, this is represented as
    a flattened `(5000,)` array. Alternative shapes include `(50, 25)` for faster computation and `(200, 100)`
    for higher-resolution results. Corresponding datasets for these three resolutions are provided.

    ## Objectives
    The goal is to optimize the distribution of solid material to minimize compliance
    (equivalently, maximize stiffness) while satisfying constraints on material usage and minimum feature size.
    Compliance is calculated as the sum of strain energy over the structure.

    The objectives are defined and indexed as follows:

    0. `c`: Compliance to minimize.

    ## Conditions
    The following input parameters define the problem conditions:
    - `volfrac`: Desired volume fraction of solid material.
    - `rmin`: Minimum feature length of beam members.
    - `forcedist`: Fractional distance of the downward force from the top-left (default) to the top-right corner.
    - `overhang_constraint`: Boolean flag to enable a 45-degree overhang constraint for manufacturability.

    ## Simulator
    Our simulation code is based on a Python adaptation of the popular 88-line topology optimization
    code. It uses the more versatile density filtering approach in combination with a standard
    Optimality Criteria (OC) optimization method. Two primary sensitivity matrices, one with respect
    to compliance (`dc`) and the other with respect to volume fraction (`dv`), are continuously updated
    and used to calculate a given design's compliance value. We have also ensured that during the
    required Lagrange multiplier search within OC, the inner optimization loop terminates if the absolute
    difference upper and lower bounds diminishes to a value smaller than machine precision. This
    prevents the code from becoming stuck at this point, which we observed in some warm-starting
    instances with noisy initial designs.

    Compliance `c` is calculated using:
    ```python
    c = ((Emin + xPrint**penal * (Emax - Emin)) * ce).sum()
    ```

    where `xPrint` is the current true density field, `penal` is the penalization factor (e.g., 3.0),
    and `ce` is the element-wise strain energy density.

    ## Dataset
    This problem offers multiple datasets for various sizes of `nelx` and `nely`. Each dataset includes
    columns for the optimal design, all conditions listed above, and the corresponding objective values.
    For advanced usage, we also provide a column containing the optimization history. The datasets have
    been generated by sampling conditions over a structured grid for various problem sizes.
    Three datasets are available on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab).
    They correspond to resolutions of $50 \times 25$, $100 \times 50$ (default), and $200 \times 100$.

    ### v0

    #### Fields
    Each dataset contains:
    - Optimized beam structures,
    - The corresponding condition parameters,
    - Objective values (compliance),
    - Full optimization histories (for advanced use).

    #### Creation Method
    Datasets were generated by uniformly sampling the condition space. The resolutions used are:
    - `(50, 25)`
    - `(100, 50)`
    - `(200, 100)`

    A more comprehensive description of the creation method can be found in the [README](https://github.com/IDEALLab/EngiBench/tree/main/engibench/problems/beams2d).

    ## Citation
    If you use this problem in your research, please cite the following paper:
    ```
    @article{andreassen2011efficient,
        title={Efficient topology optimization in MATLAB using 88 lines of code},
        author={Andreassen, Erik and Clausen, Anders and Schevenels, Mattias and Lazarov, Boyan S and Sigmund, Ole},
        journal={Structural and Multidisciplinary Optimization},
        volume={43},
        number={1},
        pages={1--16},
        year={2011},
        publisher={Springer}
    }
    ```


    ## Lead
    Arthur Drake @arthurdrake1
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection]] = (("c", ObjectiveDirection.MINIMIZE),)

    @dataclass
    class Conditions:
        """Conditions."""

        volfrac: Annotated[
            float,
            bounded(lower=0.0, upper=1.0).category(THEORY),
            bounded(lower=0.1, upper=0.9).warning().category(IMPL),
        ] = 0.35
        rmin: Annotated[
            float, greater_than(0.0).category(THEORY), bounded(lower=1.0, upper=10.0).category(IMPL).warning()
        ] = 2.0
        forcedist: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.0
        overhang_constraint: bool = False

    @dataclass
    class SimulateConfig(Conditions):
        """Common options for all workflows."""

        penal: Annotated[
            float, bounded(lower=1.0).category(IMPL), bounded(lower=2.0, upper=5.0).category(IMPL).warning()
        ] = 3.0
        nelx: Annotated[int, bounded(lower=1).category(THEORY), bounded(lower=10, upper=1000).warning().category(IMPL)] = (
            100
        )
        nely: Annotated[int, bounded(lower=1).category(THEORY), bounded(lower=10, upper=1000).warning().category(IMPL)] = 50

    @dataclass
    class Config(SimulateConfig):
        """Structured representation of configuration parameters for a numerical computation."""

        max_iter: Annotated[
            int, bounded(lower=0).category(THEORY), bounded(lower=1, upper=1000).category(IMPL).warning()
        ] = 100

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int) -> None:
            """Constraint for rmin ∈ (0.0, max{ nelx, nely }]."""
            assert 0 < rmin <= max(nelx, nely), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely)]"

    design_constraints = (volume_fraction_bound,)
    design_space = spaces.Box(low=0.0, high=1.0, shape=(Config.nely, Config.nelx), dtype=np.float64)
    dataset_id = f"IDEALLab/beams_2d_{Config.nely}_{Config.nelx}_v{version}"
    container_id = None

    def __init__(self, seed: int = 0, config: dict[str, Any] | None = None):
        """Initializes the Beams2D problem.

        Args:
            seed (int): The random seed for the problem.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.
        """
        super().__init__(seed=seed)

        # Replace the config with any new configs passed in
        self.config = self.Config(**(config or {}))

        # Restrict config to SimulateConfig fields:
        self.simulate_config = upcast(self.config)
        # Restrict further to conditions:
        self.conditions = upcast(self.simulate_config)
        self.__st = State()
        self.nelx = self.config.nelx
        self.nely = self.config.nely
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(self.nely, self.nelx), dtype=np.float64)
        self.dataset_id = f"IDEALLab/beams_2d_{self.nely}_{self.nelx}_v{self.version}"

    def simulate(
        self, design: npt.NDArray, config: dict[str, Any] | None = None, *, ce: npt.NDArray | None = None
    ) -> npt.NDArray:
        """Simulates the performance of a beam design.

        Args:
            design (np.ndarray): The design to simulate.
            ce: (np.ndarray, optional): If applicable, the pre-calculated sensitivity of the current design.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.

        Returns:
            npt.NDArray: The performance of the design in terms of compliance.
        """
        # This condition is needed to convert user-provided designs (images) to flat arrays. Normally does not apply, i.e., during optimization.
        if len(design.shape) > 1:
            design = image_to_design(design)

        simulate_config = dataclasses.replace(self.simulate_config, **(config or {}))

        # Assumes ndof is initialized as 0. This is a check to see if setup has run yet.
        # If setup has run, skips the process for repeated simulations during optimization.
        if self.__st.ndof == 0:
            self.__st = State.new(
                simulate_config.nelx, simulate_config.nely, simulate_config.rmin, simulate_config.forcedist
            )

        if ce is None:
            ce = calc_sensitivity(design, st=self.__st, cfg=dataclasses.asdict(simulate_config))
        c = (
            (self.__st.Emin + design**simulate_config.penal * (self.__st.Emax - self.__st.Emin)) * ce
        ).sum()  # compliance (objective)
        return np.array([c])

    def optimize(
        self, starting_point: npt.NDArray | None = None, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[ExtendedOptiStep]]:
        """Optimizes the design of a beam.

        Args:
            starting_point (npt.NDArray or None): The design to begin warm-start optimization from (optional).
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        base_config = self.Config(**{**dataclasses.asdict(self.simulate_config), **(config or {})})

        self.__st = State.new(base_config.nelx, base_config.nely, base_config.rmin, base_config.forcedist)

        # Returns the full history of the optimization instead of just the last step
        optisteps_history = []

        if starting_point is None:
            xPhys = base_config.volfrac * np.ones((base_config.nelx, base_config.nely), dtype=float)
            x = xPhys.ravel()
            dc = np.zeros(base_config.nely * base_config.nelx)
            dv = np.zeros(base_config.nely * base_config.nelx)
        else:
            starting_point = image_to_design(starting_point)
            assert starting_point is not None
            x = deepcopy(starting_point)
            xPhys = x.reshape((base_config.nelx, base_config.nely))
            ce = calc_sensitivity(starting_point, st=self.__st, cfg=dataclasses.asdict(base_config))
            dc = (-base_config.penal * starting_point ** (base_config.penal - 1) * (self.__st.Emax - self.__st.Emin)) * ce
            dv = np.ones(base_config.nely * base_config.nelx)

        xPrint = overhang_filter_x(xPhys) if base_config.overhang_constraint else xPhys.ravel()
        loop, change = (0, 1.0)

        while change > self.__st.min_change and loop < base_config.max_iter:
            ce = calc_sensitivity(xPrint, st=self.__st, cfg=dataclasses.asdict(base_config))
            simulate_config = upcast(base_config)
            self.reset_called = True  # override for multiple reset calls in optimize
            c = self.simulate(xPrint, ce=ce, config=dataclasses.asdict(simulate_config))

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array(c), step=loop)
            current_step.design = np.array(xPrint)
            optisteps_history.append(current_step)

            loop += 1

            dc = (-base_config.penal * xPrint ** (base_config.penal - 1) * (self.__st.Emax - self.__st.Emin)) * ce
            dv = np.ones(base_config.nely * base_config.nelx)
            # MATLAB implementation:
            if base_config.overhang_constraint:
                xPrint, dc, dv = overhang_filter_d(xPhys, dc, dv)
            else:
                xPrint = xPhys.ravel()

            dc = np.asarray(self.__st.H * (dc[np.newaxis].T / self.__st.Hs))[:, 0]
            dv = np.asarray(self.__st.H * (dv[np.newaxis].T / self.__st.Hs))[:, 0]
            # Ensure dc remains nonpositive
            dc = np.clip(dc, None, 0.0)

            xnew, xPhys, xPrint = inner_opt(x, self.__st, dc, dv, dataclasses.asdict(base_config))
            # Compute the change by the inf. norm
            change = np.linalg.norm(  # type: ignore[assignment]
                xnew.reshape(base_config.nelx * base_config.nely, 1) - x.reshape(base_config.nelx * base_config.nely, 1),
                np.inf,
            )
            x = deepcopy(xnew)

        return design_to_image(xPrint, base_config.nelx, base_config.nely), optisteps_history

    def reset(self, seed: int | None = None, **kwargs) -> None:
        r"""Reset numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        super().reset(seed, **kwargs)
        self.__st = State()

    def render(self, design: np.ndarray, *, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(design, cmap="coolwarm", ax=ax, vmin=0, vmax=1)

        if open_window:
            plt.show()
        return fig, ax

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
    # Provides a way to instantiate the problem without having to pass configs to optimize or simulate later.
    # Possible sets of nely and nelx: (25, 50), (50, 100), and (100, 200)
    # If a new nely and nelx are not passed in, uses the default conditions.

    problem = Beams2D(seed=0)

    print(f"Loading dataset for nely={problem.nely}, nelx={problem.nelx}.")
    dataset = problem.dataset

    # Example of getting the training set
    optimal_train = dataset["train"]["optimal_design"]
    c_train = dataset["train"]["c"]
    condition_keys = [f.name for f in dataclasses.fields(problem.Conditions)]
    params_train = dataset["train"].select_columns(condition_keys)

    # Get design and conditions from the dataset, render design
    # Note that here, we override any previous configs to re-optimize the same design as a test case.
    design, idx = problem.random_design()
    config = params_train[idx]
    compliance = c_train[idx]
    fig, ax = problem.render(design, open_window=True)

    print(f"Verifying compliance via simulation. Reference value: {compliance:.4f}")

    try:
        c_ref = problem.simulate(design, config=config)[0]
        print(f"Calculated compliance: {c_ref:.4f}")
    except ArithmeticError:
        print("Failed to calculate compliance for upscaled design.")

    # Sample Optimization
    print("\nNow conducting a sample optimization with the given configs:", config)
    problem.reset(seed=1)

    # NOTE: optimal_design and optisteps_history[-1].stored_design are interchangeable.
    optimal_design, optisteps_history = problem.optimize(config=config)
    print(f"Final compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(f"Final design volume fraction: {optimal_design.sum() / (np.prod(optimal_design.shape)):.4f}")

    fig, ax = problem.render(optimal_design, open_window=True)
