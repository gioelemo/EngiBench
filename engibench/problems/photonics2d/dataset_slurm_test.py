"""Dataset Generation for Photonics2D Problem via SLURM.

This script generates a dataset for the Photonics2D problem using the SLURM API
"""

from argparse import ArgumentParser
from itertools import product
import os
import time
from typing import Any

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm


def run_problem(config: dict[str, Any], fig_path: str, *, simulate: bool, **optimize_args: Any):
    """Create and optimize a single problem."""
    problem = Photonics2D(**config)
    start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
    final_design, _obj_trajectory = problem.optimize(start_design, **optimize_args)
    fig = problem.render(design=final_design, config=optimize_args)
    fig.savefig(fig_path)
    if simulate:
        problem.simulate(final_design)


def run_slurm(configs: list[dict], fig_dir: str, *, simulate: bool, **optimize_config: Any) -> None:
    """Function to optimize designs via SLURM."""
    os.makedirs(fig_dir, exist_ok=True)
    # Make slurm Args
    parameter_space = [
        {
            "config": config,
            "simulate": simulate,
            "fig_path": os.path.join(fig_dir, f"final_design_{index}.png"),
            **optimize_config,
        }
        for index, config in enumerate(configs)
    ]
    print(f"Generating parameter space via SLURM with {len(parameter_space)} configurations.")

    # --------- Testing `optimize` via SLURM ---------
    # First let's check if we can run `optimize``
    print("Starting `pipeline` via SLURM...")
    start_time = time.time()
    slurm.sbatch_map(
        run_problem,
        args=parameter_space,
        slurm_args=slurm.SlurmConfig(log_dir="./opt_logs/", runtime=runtime_optimize),
        wait=True,
    )
    end_time = time.time()
    print(f"Elapsed time for `optimize`: {end_time - start_time:.2f} seconds")


# If we just want to render the designs locally, we can do that here. If rendering takes a while
# then we might want to run it via SLURM, which is later in this script. Below is just for
# local execution of render.
def render_local(target_problem: Any, opt_results: list[dict], fig_path: str) -> None:
    """Function to render designs locally, without SLURM."""
    # Check if `figs` directory exists, if not create it
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for _i, result in enumerate(opt_results):
        final_design, _obj_trajectory = result["results"]
        problem = target_problem(result["problem_args"])
        fig = problem.render(design=final_design, config=result["simulate_args"])
        fig.savefig(fig_path + f"/final_design_{_i}.png")


if __name__ == "__main__":
    """Dataset Generation, Optimization, Simulation, and Rendering for Photonics2D Problem via SLURM.

    This script generates a dataset for the Photonics2D problem using the SLURM API, though it could
    be generalized to other problems as well. It includes functions for optimization, simulation,
    and rendering of designs.

    Command Line Arguments:
    -r, --render: Should we render the optimized designs?
    --figure_path: Where should we place the figures?
    -s, --simulate: Should we simulate the optimized designs?

    """
    # Fetch command line arguments for render and simulate to know whether to run those functions
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        dest="render_flag",
        default=False,
        help="Should we render the optimized designs?",
    )
    parser.add_argument("--figure_path", dest="fig_path", default="./figs", help="Where should we place the figures?")
    parser.add_argument(
        "-s",
        "--simulate",
        action="store_true",
        dest="simulate_flag",
        default=False,
        help="Should we simulate the optimized designs?",
    )
    args = parser.parse_args()

    # ============== Problem-specific elements ===================
    # The following elements are specific to the problem and should be modified accordingly
    target_problem = Photonics2D
    # Specify the parameters you want to sweep over for optimization
    rng = np.random.default_rng()
    lambda1 = rng.uniform(low=0.5, high=1.25, size=20)
    lambda2 = rng.uniform(low=0.75, high=1.5, size=20)
    blur_radius = range(5)
    num_elems_x = 120
    num_elems_y = 120

    # Generate all combinations of parameters to run
    combinations = list(product(lambda1, lambda2, blur_radius))

    # Generate full problem configurations, including static parameters
    # Note that currently this doesn't allow you to change the resolution of the problem
    # So in the re-write of the SLURM API we will need to add that functionality.
    def config_factory(lambda1: float, lambda2: float, blur_radius: int) -> dict:
        """Factory function to create configuration dictionaries."""
        return {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "blur_radius": blur_radius,
        }

    # Call the config factory ro generate configurations
    configs = [config_factory(l1, l2, br) for l1, l2, br in combinations]

    # Any optimization configurations can be set here, if you want
    optimize_config = {"num_optimization_steps": 200}

    # Timing information for `optimize` and `simulate` functions for SLURM
    # If you can estimate the time it takes to run `optimize` and `simulate`,
    # you can set the runtimes here, and this will help with job scheduling.
    # Try to be conservative with the time estimates, so SLURM doesn't kill it prematurely.
    # The format is "HH:MM:SS"
    runtime_optimize = "00:12:00"  # ~10 minutes for optimization
    runtime_simulate = "00:02:00"  # ~1 minute for simulation
    runtime_render = "00:02:00"  # ~1 minutes for rendering

    # ============== End of problem-specific elements ===================

    # Now call optimize
    run_slurm(configs, optimize_config=optimize_config, fig_dir=args.fig_path, simulate=args.simulate_flag)
