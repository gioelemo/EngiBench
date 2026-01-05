"""C-TAEA Optimization for 2D Truss Structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

# --- CHANGED: Import C-TAEA and Reference Directions ---
from pymoo.algorithms.moo.ctaea import CTAEA

# -------------------------------------------------------
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from engibench.problems.truss2d.model import utils
from engibench.problems.truss2d.optimize.node_sort import search_algorithm

if TYPE_CHECKING:
    from engibench.problems.truss2d.v0 import Truss2D


class Truss2dCTAEA:
    """C-TAEA optimization for 2D truss structures."""

    def __init__(self, truss2d: Truss2D, initial_designs: npt.NDArray, population_size: int = 100,
                 generations: int = 100, *, node_sort_init: bool = False) -> None:
        self.truss2d = truss2d
        self.conditions = truss2d.conditions
        self.population_size = population_size
        self.generations = generations
        self.n_binary_variables = utils.get_num_bits(self.conditions)
        self.rng = np.random.default_rng()
        self.node_sort_init = node_sort_init

        # Calculate Reference Point (Worst Case)
        design_all_members = [1 for _ in range(self.n_binary_variables)]
        _, self.volume_ref, _ = self.evaluate_design(design_all_members)
        self.stiffness_ref = 0.0

        self.initial_designs = initial_designs

    def evaluate_design(self, design: list[int]) -> tuple[float, float, float]:
        """Evaluate a design and return its objectives and constraints."""
        results = self.truss2d.simulate(design)
        stiffness = results["stiffness_avg"]
        volume = results["volume"]
        constraint_score = results["member_overlaps"]

        # Soft penalty for invalid designs
        if constraint_score > 0:
            stiffness = 0.0
        return stiffness, volume, constraint_score

    def solve(self):
        """Executes the C-TAEA optimization."""
        # 1. Handle Smart Population Initialization
        sampling = self.init_population()

        # 2. Setup Reference Directions (REQUIRED for C-TAEA)
        # We generate linear reference directions for 2 objectives.
        # n_partitions is set to match population_size closely.
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.population_size - 1)

        # 3. Setup Algorithm
        algorithm = CTAEA(
            ref_dirs=ref_dirs,  # Passed here instead of pop_size
            sampling=sampling,
            crossover=TwoPointCrossover(),
            mutation=IntegerBitFlipMutation(),
            eliminate_duplicates=True
        )

        problem = TrussProblem(self)

        # Reference point for Hypervolume callback
        ref_point = np.array([self.volume_ref * 1.1, self.stiffness_ref])

        print("Starting C-TAEA Optimization...")
        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.generations),
            callback=HVCallback(ref_point),
            seed=1,
            verbose=False
        )

        if res.X is None or res.F is None or res.algorithm is None:
            raise ValueError("Optimization failed to produce results.")

        # Extract results
        final_volumes = res.F[:, 0]
        final_stiffness = -res.F[:, 1]

        print("Designs found:", len(res.X))

        return {
            "X": res.X.astype(int),
            "Volume": final_volumes,
            "Stiffness": final_stiffness,
            "Hypervolume_History": res.algorithm.callback.history
        }

    def init_population(self):
        """Initializes the population with node_sort seeding."""
        problem = TrussProblem(self)
        sampling = BinaryRandomSampling()

        if self.node_sort_init:
            print("Generating node-sort initial designs...")
            node_sort_designs = search_algorithm(conditions=self.conditions, load_idx=0)

            # Manual append example (ensure this logic matches your domain needs)
            temp_design = [[0, 3], [3, 11], [8, 11], [0, 6], [3, 6], [6, 8], [6, 11]]
            temp_design, _, _, _ = utils.convert(self.conditions, temp_design)
            node_sort_designs.append(temp_design)

            if self.initial_designs.shape[0] == 0:
                self.initial_designs = node_sort_designs
            else:
                self.initial_designs = np.vstack([self.initial_designs, node_sort_designs])

        if self.initial_designs is not None and len(self.initial_designs) > 0:
            x_seed = np.array(self.initial_designs)
            n_seeds = len(x_seed)

            if n_seeds > self.population_size:
                print(f"Seeding with {n_seeds} designs. Running survival to select best {self.population_size}...")

                # Note: We use NSGA2's RankAndCrowdingSurvival for the pre-filter
                # because it is efficient at reducing a large seed set to a fixed size.
                pop = Population.new("X", x_seed)
                Evaluator().eval(problem, pop)
                survivors = RankAndCrowdingSurvival().do(problem, pop, n_survive=self.population_size)

                if survivors is None:
                    raise ValueError("Failed to evaluate seed population.")
                sampling = survivors.get("X")

            elif n_seeds < self.population_size:
                print(f"Seeding with {n_seeds} designs and padding with random individuals.")
                n_random = self.population_size - n_seeds
                x_random = self.rng.integers(0, high=2, size=(n_random, self.n_binary_variables))
                sampling = np.vstack([x_seed, x_random])
            else:
                print(f"Seeding with exactly {self.population_size} provided designs.")
                sampling = x_seed

        return sampling


class TrussProblem(ElementwiseProblem):
    """Pymoo Problem Wrapper for 2D Truss Optimization."""

    def __init__(self, outer_instance):
        super().__init__(
            n_var=outer_instance.n_binary_variables,
            n_obj=2,
            n_ieq_constr=1,
            xl=0,
            xu=1,
            vtype=int
        )
        self.outer = outer_instance

    def _evaluate(self, x, out, *_args, **_kwargs):
        design_bitlist = x.astype(int).tolist()
        stiffness, volume, constraint_score = self.outer.evaluate_design(design_bitlist)
        out["F"] = [volume, -stiffness]
        out["G"] = [constraint_score]


class IntegerBitFlipMutation(Mutation):
    """Bit Flip Mutation for Integer Binary Vectors."""

    def __init__(self, prob=None, rng=None):
        super().__init__()
        self.prob = prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def _do(self, problem, x, **_kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var
        xp = np.copy(x)
        flip = self.rng.random(x.shape) < self.prob
        xp[flip] = 1 - x[flip]
        return xp


class HVCallback(Callback):
    """Callback to compute and log Hypervolume at each generation."""

    def __init__(self, ref_point):
        super().__init__()
        self.hv_indicator = Hypervolume(ref_point=ref_point)
        self.history = []

    def notify(self, algorithm):
        """Called at each generation to compute hypervolume."""
        f = algorithm.opt.get("F")
        if f is not None and len(f) > 0:
            hv_value = self.hv_indicator(f)
            self.history.append(hv_value)
            print(f"Gen {algorithm.n_gen} | Hypervolume: {hv_value:.4f} | Archive Size: {len(f)}")
