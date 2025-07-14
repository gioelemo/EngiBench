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
from engibench.problems.airfoil.simulation_jobs import simulate_slurm, generate_configuration_args_for_optimize, post_process_simulate
from datasets import load_dataset
print(f"Python version: {sys.version}")



if __name__ == "__main__":
    """Dataset Generation, Simulation, and Rendering for Airfoil Problem via SLURM.

    This script generates a dataset for the Airfoil problem using the SLURM API, though it could
    be generalized to other problems as well. It includes functions for simulation,
    and rendering of designs.

    Command Line Arguments:
    -r, --render: Should we render the optimized designs?
    --figure_path: Where should we place the figures?
    -s, --simulate: Should we simulate the optimized designs?

    """
    # Fetch command line arguments for render and simulate to know whether to run those functions
    #parser = ArgumentParser()
    #parser.add_argument(
    #    "-r",
    #    "--render",
    #    action="store_true",
    #    dest="render_flag",
    #    default=False,
    #    help="Should we render the optimized designs?",
    #)
    #parser.add_argument("--figure_path", dest="fig_path", default="./figs", help="Where should we place the figures?")
    #parser.add_argument(
    #    "-s",
    #    "--simulate",
    #    action="store_true",
    #    dest="simulate_flag",
    #    default=False,
    #    help="Should we simulate the optimized designs?",
    #)
    #args = parser.parse_args()

    # ============== Problem-specific elements ===================
    # The following elements are specific to the problem and should be modified accordingly
    # Specify the parameters you want to sweep over for optimization
    rng = np.random.default_rng()
    params_to_sweep = {
        "mach": rng.uniform(low=0.5, high=0.9, size=1),
        "reynolds": rng.uniform(low=1.0e6, high=2.0e7, size=1),
        "alpha": rng.uniform(low=0.0, high=20.0, size=1),
    }

    # For the updated SLURM API, we need to define four things:
    # 1. The job that we want SLURM to run -- this is essentially a factory for different
    #    problem runs and should take in the arguments that we want to vary.
    # 2. An `args` variable, which is all of the arguments that we want to pass to the job.
    #    SLURM will run `job` with each of the arguments in `args`.
    # 3. A function that does the post-processing of the results from the job. This gets passed
    #    via `reduce_job` and can also handle any JobErrors that occur.
    # 4. Lastly, any SLURM Configurations, including runtime, memory, and the group_size

    # Generate params
    simulate_configurations = generate_configuration_args_for_optimize(params_to_sweep)
    simulate_configs_designs = []
    
    # Generate designs
    ds = load_dataset("IDEALLab/airfoil_v0")
    designs = ds["train"]["initial_design"]
    problem_configuration = {'mach': np.float16(0.05), 'reynolds': np.float16(200), 'alpha': np.float16(10.0)}
    for i, design in enumerate(designs):
        config = {'problem_configuration': problem_configuration, 'configuration_id': i}
        config["design"] = designs[i]["coords"]
        simulate_configs_designs.append(config)
    
    slurm_config = slurm.SlurmConfig(
        name="Airfoil_dataset_generation",
        runtime="00:10:00",  # Give 15 minutes for each simulation job
        log_dir="./sim_logs/",
    )
    slurm.sbatch_map(
        f=simulate_slurm,
        args=simulate_configs_designs,
        slurm_args=slurm_config,
        group_size=2,  # Number of jobs to batch in sequence to reduce job array size
        reduce_job=post_process_simulate,
        out="results.pkl",
    )
    #results = slurm.load_results()
    #for result in results:
    #    if isinstance(result, slurm.JobError):
    #        raise result