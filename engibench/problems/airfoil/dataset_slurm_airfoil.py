"""Dataset Generator for the Photonics2D problem using the updated SLURM API."""

from argparse import ArgumentParser
from itertools import product
import os, sys
import shutil
import time
from typing import Any
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import qmc

from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.v0 import Airfoil
from engibench.utils import slurm
from engibench.problems.airfoil.simulation_jobs import simulate_slurm, generate_configuration_args_for_optimize, post_process_simulate
from datasets import load_dataset
print(f"Python version: {sys.version}")


def viz_dists(x, c, color):
    x_labels = ['X-Coordinates', 'Y-Coordinates']
    c_labels = ['Mach Number', 'Reynolds Number', 'Angle of Attack']
    for i in range(2):
        plt.subplot(3, 3, i+1)
        data = x[:, i, :].flatten()  # Added .cpu()
        plt.hist(data, bins=50, alpha=0.5, edgecolor='black', color=color, density=False)
        plt.title(x_labels[i])
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    # Process tensor2: [35000, 2]
    for i in range(3):
        plt.subplot(3, 3, i+4)
        data = c[:, i]
        plt.hist(data, bins=50, alpha=0.5, edgecolor='black', color=color, density=False)
        plt.title(c_labels[i])
        plt.xlabel('Value')
        plt.ylabel('Frequency')


    plt.tight_layout()
    plt.show()

    return None

def calculate_runtime(group_size, minutes_per_sim=5):
    total_minutes = group_size * minutes_per_sim
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}:00"

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
    #params_to_sweep = {
    #    "mach": rng.uniform(low=0.5, high=0.9, size=100),
    #    "reynolds": rng.uniform(low=1.0e6, high=2.0e7, size=100),
    #    "alpha": rng.uniform(low=0.0, high=20.0, size=50),
    #}
    

    # For the updated SLURM API, we need to define four things:
    # 1. The job that we want SLURM to run -- this is essentially a factory for different
    #    problem runs and should take in the arguments that we want to vary.
    # 2. An `args` variable, which is all of the arguments that we want to pass to the job.
    #    SLURM will run `job` with each of the arguments in `args`.
    # 3. A function that does the post-processing of the results from the job. This gets passed
    #    via `reduce_job` and can also handle any JobErrors that occur.
    # 4. Lastly, any SLURM Configurations, including runtime, memory, and the group_size

    # Generate params
    #simulate_configurations = generate_configuration_args_for_optimize(params_to_sweep)
    simulate_configs_designs = []
    
    # Generate designs
    ds = load_dataset("IDEALLab/airfoil_v0")
    designs = ds["train"]["initial_design"]+ds["train"]["optimal_design"]+\
              ds["val"]["initial_design"]+ds["val"]["optimal_design"]+\
              ds["test"]["initial_design"]+ds["test"]["optimal_design"]
    n_designs = len(designs)

    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=2)
    samples = sampler.random(n=n_designs)  # n samples needed

    # Scale to your domain
    bounds = np.array([[0.5, 0.9], [1.0e6, 2.0e7]])
    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    mach_values = scaled_samples[:, 0]
    reynolds_values = scaled_samples[:, 1]

    config_id = 0
    for i, (design, ma, re) in enumerate(zip(designs, mach_values, reynolds_values)):
        for j, alpha in enumerate(rng.uniform(low=0.0, high=20.0, size=50)):

            problem_configuration = {'mach': ma, 'reynolds': re, 'alpha': alpha}
            config = {'problem_configuration': problem_configuration, 'configuration_id': config_id}
            config["design"] = design["coords"]
            simulate_configs_designs.append(config)
            config_id += 1




    simulate_configs_designs = np.random.permutation(simulate_configs_designs).tolist()

    # Set up the figure
    fig = plt.figure(figsize=(15, 10))

    # Concatenate train/test
    mach = [config['problem_configuration']['mach'] for config in simulate_configs_designs]
    reynolds = [config['problem_configuration']['reynolds'] for config in simulate_configs_designs]
    alpha = [config['problem_configuration']['alpha'] for config in simulate_configs_designs]
    coords = [config['design'] for config in simulate_configs_designs]

    x = np.array(coords)
    c = np.array([mach, reynolds, alpha]).T
    print(x.shape, c.shape)

    viz_dists(x, c, 'blue')
    plt.savefig('study_histograms.png')

    print(f"Generated {len(simulate_configs_designs)} configurations for simulation.")
    
    n_simulations = len(simulate_configs_designs)
    n_slurm_array = 1000
    group_size = 12
    n_sbatch_maps = np.ceil(n_simulations / (group_size * n_slurm_array))

    slurm_config = slurm.SlurmConfig(
        name="Airfoil_dataset_generation",
        runtime=calculate_runtime(group_size, minutes_per_sim=5),
        ntasks=1,
        cpus_per_task=1,
        log_dir="./sim_logs/",
    )
    print(calculate_runtime(group_size, minutes_per_sim=5))

    submitted_jobs = []
    for ibatch in range(int(n_sbatch_maps)):
        sim_batch_configs = simulate_configs_designs[ibatch * group_size * n_slurm_array: (ibatch + 1) * group_size * n_slurm_array]
        print(len(sim_batch_configs))
        print(f"Submitting batch {ibatch + 1}/{int(n_sbatch_maps)}")

        job_array = slurm.sbatch_map(
            f=simulate_slurm,
            args=sim_batch_configs,
            slurm_args=slurm_config,
            group_size=group_size,  # Number of jobs to batch in sequence to reduce job array size
            reduce_job=post_process_simulate,
            out=None,
        )

        # Save the job array reference
        submitted_jobs.append(job_array)

        # Wait for this job to complete by calling save()
        # This will submit a dependent job that waits for the array to finish
        print(f"Waiting for batch {ibatch + 1} to complete...")
        job_array.save(f"results_{ibatch}.pkl", slurm_args=slurm_config)
        print(f"Batch {ibatch + 1} completed!")

    #results = slurm.load_results()
    #for result in results:
    #    if isinstance(result, slurm.JobError):
    #        raise result