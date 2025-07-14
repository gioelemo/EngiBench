"""Dataset Generator for the Photonics2D problem using the updated SLURM API."""

from argparse import ArgumentParser
from itertools import product
import os, sys
import shutil
import time
from typing import Any

import numpy as np

from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.v0 import Airfoil
from engibench.utils import slurm

def compute_total_elapsed_time(return_values: list[dict]) -> float:
    """Computes the total elapsed time (in seconds) across all simulations."""
    total_elapsed_time = 0
    for result in return_values:
        # Retrieve all of the result elements
        optimized_design, opti_history, elapsed_time, problem_configuration, configuration_id = result

        # 1. Compute the total elapsed time across all simulations
        total_elapsed_time += elapsed_time
    return total_elapsed_time


'''
def render_all_final_designs(return_values: list[dict], fig_path: str = "result_figures") -> None:
    """Renders all final designs and saves them in a zip file for easy download."""
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for _i, result in enumerate(return_values):
        # Retrieve all of the result elements
        optimized_design, opti_history, elapsed_time, problem_configuration, configuration_id = result
        problem = Photonics2D(problem_configuration)
        fig = problem.render(design=optimized_design)
        fig.savefig(fig_path + f"/final_design_{_i}.png")
        # Now zip the figure directory
    zip_filename = "figures_all"
    shutil.make_archive(zip_filename, "zip", fig_path)
    print(f"Saved image archive in {zip_filename}.")
'''


def post_process_simulate(return_values: list[dict]):
    """Post-Processing script that can operate on all of the returned results."""
    # In this case, the main purpose of this post-processing step could be two fold:
    # 1. Computing the total time for dataset generation and returning this.
    # 2. Computing all of the final rendered designs and storing this in a zip file
    #    so that users could get images of all the final designs.
    #
    total_elapsed_time = compute_total_elapsed_time(return_values)
    print(f"Total Elapsed time (in seconds): {total_elapsed_time}")

    render_all_final_designs(return_values, "result_figures")


def dict_cartesian_product(d):
    """Generates the cartesian product of a dictionary of lists."""
    keys = d.keys()
    values_product = product(*d.values())
    return [dict(zip(keys, values, strict=True)) for values in values_product]


def generate_configuration_args_for_optimize(params_to_sweep: dict[str, Any]) -> list[dict]:
    """Takes in a Problem and a lists of parameters to sweep, and generates the cartesian product of those params."""
    # Generate all combinations of each parameter across params_to_sweep
    combinations = dict_cartesian_product(params_to_sweep)

    # Now add in a configuration_id to each of the combinations, and assemble this into an args list
    # This is just a unique identifier for each of the combinations, in case you want to debug a given run
    args = []
    for i, config in enumerate(combinations):
        args.append(
            {
                "problem_configuration": config,
                "configuration_id": i,
            }
        )
    return args


def simulate_slurm(problem_configuration: dict, configuration_id: int, design: dict) -> dict:
    """Takes in the given problem, configuration, and designs, then runs the optimization.

    Any arguments should be things that you want to change across the different jobs, and anything
    that is the same/static across the runs should just be defined inside this function.

    Args:
        problem (Problem): The problem to run the optimization on.
        problem_configuration (dict): The specific configuration used to setup the problem being passed.
            This parameter is just being passed through for reporting, since problem should already be
            configured with these settings.
        configuration_id (int): A unique identifier for the job for later debugging or tracking.

    Returns:
        "simulated_design": The simulated design.
        "simulated_fields": The simulated fields for the design.
        "simulate_time": The time taken to run this simulation job. Useful for aggregating
            the time taken for dataset generation.
    """
    #base_directory = 'scratch/workdir/'+'study_' + str(configuration_id)
    #print(base_directory)
    #os.makedirs(base_directory, exist_ok=True)
    problem = Airfoil()
    sim_id = configuration_id+1
    problem.reset(seed=sim_id, cleanup=False)
    my_design = {"coords": np.array(design), "angle_of_attack": problem_configuration["alpha"]}

    print("Starting `simulate` via SLURM...")
    start_time = time.time()
    performance, fields = problem.simulate(my_design, mpicores=1, config=problem_configuration)
    performance_fields_dict = {'drag': performance[0], 'lift': performance[1], 'x-coordinates': fields[0], 'y-coordinates': fields[1], 'pressure': fields[2], 'x-velocity': fields[3], 'y-velocity': fields[4]}
    #print(performance_fields_dict)
    print("Finished `simulate` via SLURM.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for `simulate`: {elapsed_time:.2f} seconds")

    return {
        "performance_dict": performance_fields_dict,
        "optimize_time": elapsed_time,
        "problem_configuration": problem_configuration,
        "configuration_id": configuration_id,
    }