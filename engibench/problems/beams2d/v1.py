# ruff: noqa: N806
# Disabled variable name conventions

"""Beams 2D problem."""

from copy import deepcopy
import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import image_to_design
from engibench.problems.beams2d.backend import inner_opt
from engibench.problems.beams2d.backend import overhang_filter_d
from engibench.problems.beams2d.backend import overhang_filter_x
from engibench.problems.beams2d.backend import State
from engibench.problems.beams2d.v0 import Beams2D as Beams2D_v0
from engibench.problems.beams2d.v0 import ExtendedOptiStep
from engibench.utils.upcast import upcast


class Beams2D(Beams2D_v0):
    r"""Beam 2D topology optimization problem - Version 1 (v1).

    ### v1
    This version augments v0 by fixing a minor detail in the v0 warm-start optimization process.
    Specifically, when warm-starting from a provided design, a small epsilon value is added to
    avoid zero-density values that could lead to gradient issues. The datasets themselves remain unchanged.

    All other behavior is identical to v0.
    See v0.py for full baseline documentation.
    """

    version = 1

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
        else:
            starting_point = image_to_design(starting_point)
            assert starting_point is not None
            eps = 1e-4
            x = (
                (1 - eps) * starting_point + eps * base_config.volfrac
            )  # add tiny non-zero values to avoid warm-start gradient issues for zero values
            xPhys = x.reshape((base_config.nelx, base_config.nely))

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
