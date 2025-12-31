"""NSGA-II optimization for 2D truss structures using EngiBench framework."""



from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from engibench.problems.truss2d.model import utils
from engibench.problems.truss2d.optimize.node_sort import search_algorithm

if TYPE_CHECKING:
    from engibench.problems.truss2d.v0 import Truss2D




class Truss2dNSGA2:
    """NSGA-II optimization for 2D truss structures."""

    def __init__(self, truss2d: Truss2D, initial_designs: npt.NDArray | None = None, population_size: int = 100, generations: int = 100) -> None:
        self.truss2d = truss2d
        self.conditions = truss2d.conditions
        self.population_size = population_size
        self.generations = generations
        self.n_binary_variables = utils.get_num_bits(self.conditions)
        self.rng = np.random.default_rng()  # Centralized generator

        # Calculate Reference Point (Worst Case)
        # Obj 1 (Volume): Worst case is all members present.
        # Obj 2 (Stiffness): Worst case is 0 stiffness (which we negate to -0.0)
        design_all_members = [1 for _ in range(self.n_binary_variables)]
        _,  self.volume_ref, _ = self.evaluate_design(design_all_members)
        self.stiffness_ref = 0.0

        # Generate Initial Designs (Optional)
        self.initial_designs = initial_designs
        self.initial_designs = search_algorithm(conditions=self.conditions, load_idx=0)


    def evaluate_design(self, design: list[int]) -> tuple[float, float, float]:
        """Evaluate a design and return its objectives and constraints.

        Args:
            design (list[int]): The design represented as a bit list.

        Returns:
            stiffness (float): The stiffness of the design (to be maximized).
            volume (float): The volume of the design (to be minimized).
            constraint_score (float): The constraint score (to be minimized, 0 if feasible).
        """
        results = self.truss2d.simulate(design)
        stiffness = results["stiffness_avg"]           # Maximize
        volume = results["volume"]                     # Minimize
        constraint_score = results["member_overlaps"]  # Minimize (0 when design is feasible)
        if constraint_score > 0:
            stiffness = 0.0
        return stiffness, volume, constraint_score

    def solve( # noqa: C901
            self):
        """Executes the constrained NSGA-II optimization.

        Args:
            initial_designs: A list of bit lists to seed the population.
        """

        # 1. Define the Pymoo Problem Wrapper (Same as before)
        class TrussProblem(ElementwiseProblem):
            def __init__(self, outer_instance):
                super().__init__(
                    n_var=outer_instance.n_binary_variables,
                    n_obj=2,
                    n_ieq_constr=1,
                    xl=0,
                    xu=1,
                    vtype=int  # <--- CRITICAL FIX: Tells pymoo these are discrete
                )
                self.outer = outer_instance

            def _evaluate(self, x, out, *_args, **_kwargs):
                design_bitlist = x.astype(int).tolist()
                stiffness, volume, constraint_score = self.outer.evaluate_design(design_bitlist)
                out["F"] = [volume, -stiffness]
                out["G"] = [constraint_score]

        # 2. Instantiate Problem immediately (needed for pre-evaluation)
        problem = TrussProblem(self)

        # 3. Handle Smart Initialization
        sampling = BinaryRandomSampling()  # Default if no designs provided

        if self.initial_designs is not None and len(self.initial_designs) > 0:
            x_seed = np.array(self.initial_designs)
            n_seeds = len(x_seed)

            if n_seeds > self.population_size:
                print(f"Seeding with {n_seeds} designs. Running survival to select best {self.population_size}...")

                # A. Create Population
                pop = Population.new("X", x_seed)

                # B. Use Evaluator (THE FIX)
                # Evaluator extracts 'X' from pop, evaluates it, and saves 'F', 'G', 'CV' back to pop
                Evaluator().eval(problem, pop)

                # C. Run Survival
                survivors = RankAndCrowdingSurvival().do(problem, pop, n_survive=self.population_size)

                if survivors is None:
                    raise ValueError("Failed to evaluate seed population.")

                # D. Extract survivors for sampling
                sampling = survivors.get("X")

            elif n_seeds < self.population_size:
                # Padding logic for when we have too few designs
                print(f"Seeding with {n_seeds} designs and padding with random individuals.")
                n_random = self.population_size - n_seeds
                x_random = self.rng.integers(0, high=2, size=(n_random, self.n_binary_variables))
                sampling = np.vstack([x_seed, x_random])
            else:
                # Exact match
                print(f"Seeding with exactly {self.population_size} provided designs.")
                sampling = x_seed

        # 4. Define Callback
        class HVCallback(Callback):
            def __init__(self, ref_point):
                super().__init__()
                self.hv_indicator = Hypervolume(ref_point=ref_point)
                self.history = []

            def notify(self, algorithm):
                f = algorithm.opt.get("F")
                if f is not None and len(f) > 0:
                    hv_value = self.hv_indicator(f)
                    self.history.append(hv_value)
                    print(f"Gen {algorithm.n_gen} | Hypervolume: {hv_value:.4f} | Archive Size: {len(f)}")

        # 4. Setup Reference Point and Algorithm
        ref_point = np.array([self.volume_ref * 1.1, self.stiffness_ref])

        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=sampling,  # Pass our constructed matrix here
            crossover=TwoPointCrossover(),
            mutation=IntegerBitFlipMutation(),
            eliminate_duplicates=True
        )

        problem = TrussProblem(self)

        print("Starting NSGA-II Optimization...")
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

        final_volumes = res.F[:, 0]
        final_stiffness = -res.F[:, 1]

        print("Designs found:", len(res.X))

        return {
            "X": res.X.astype(int),
            "Volume": final_volumes,
            "Stiffness": final_stiffness,
            "Hypervolume_History": res.algorithm.callback.history
        }




class IntegerBitFlipMutation(Mutation):
    """Bit Flip Mutation for Integer Binary Vectors."""

    def __init__(self, prob=None, rng=None):
        super().__init__()
        self.prob = prob
        # Use provided generator or create a new default one
        self.rng = rng if rng is not None else np.random.default_rng()

    def _do(self, problem, x, **_kwargs):
        """Applies bit flip mutation to the population X."""
        # Default probability is 1/n_var if not provided
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        # Create mutation mask
        xp = np.copy(x)

        # FIX: Use self.rng.random instead of np.random.random
        flip = self.rng.random(x.shape) < self.prob

        # ARITHMETIC FLIP: 1 - 0 = 1; 1 - 1 = 0
        xp[flip] = 1 - x[flip]

        return xp
