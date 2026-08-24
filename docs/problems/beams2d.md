# Beams2D

``` {problem:table}
:lead: Arthur Drake @arthurdrake1
```

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

```{problem:conditions}
```

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
and `ce` is the element-wise strain energy density. This expression is the same in every version,
so objective values are directly comparable across versions and across datasets.

### Optimizer (v2, AutoSiMP)
Since v2 the optimizer follows the solver-side approach of AutoSiMP
([arXiv:2603.27000](https://arxiv.org/abs/2603.27000)). The plain density filter of v0/v1 is replaced
by a **three-field** parameterization: the design field $x$ is filtered into $\tilde{x} = Hx/H_s$ and
then pushed through a smooth Heaviside projection

$$\bar{x} = \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde{x}-\eta))}
                 {\tanh(\beta\eta) + \tanh(\beta(1-\eta))},$$

and it is the projected field $\bar{x}$ that enters the stiffness assembly, the volume constraint and
the returned design. Compliance and volume sensitivities are chained back through the projection and
the filter, so the optimizer stays exactly consistent with the physics it reports.

Three further AutoSiMP components come with it:

* **Pluggable continuation control.** At every iteration a controller receives an `Observation`
  (compliance, best compliance, grayness, volume fraction, checkerboard index, stagnation counter,
  budget consumption) and returns the four *Direct Numeric Control* parameters: the penalization
  exponent $p$, the projection sharpness $\beta$, the filter radius $r_{\min}$ and the optimality
  criteria move limit $\delta$. `continuation="schedule"` (the default) is the deterministic,
  budget-aware continuation; `continuation="fixed"` is the no-continuation baseline; and any callable
  can be handed to `optimize(controller=...)`, which is where an LLM agent plugs in.
* **Eight-check structural evaluator**, described below.
* **Closed-loop retry.** When a gating check fails, the solver escalates its own settings (filter
  radius, sharpness cap, move limit, iteration budget) and re-runs, up to `max_retries` times, keeping
  the best attempt.

The two remaining AutoSiMP modules -- the LLM configurator that parses a plain-English prompt and the
boundary-condition generator that turns the resulting specification into solver arrays -- are not
reimplemented here: the validated specification they produce is precisely what EngiBench's
`Conditions` dataclass already is, and EngiBench does not depend on an LLM provider.

## Quality Evaluation
`optimize` runs the eight-check structural evaluator on its result and stores the outcome in
`problem.last_quality_report` (one report per attempt is kept in `problem.quality_reports`). Any
design can also be checked directly with `problem.evaluate_quality(design)`.

Five checks gate a run:

| Check | Criterion | Default threshold |
| --- | --- | --- |
| `connectivity` | a 4-neighbour flood fill on the thresholded design links a loaded node to a constrained node | — |
| `compliance_ratio` | final compliance over the best compliance seen (tail degradation) | `< 2.0` |
| `grayness` | non-discreteness $M_{nd} = \frac{1}{n}\sum_e 4x_e(1-x_e)$ | `<= 0.15` |
| `volume_fraction` | relative deviation from the target volume fraction | `<= 2%` |
| `convergence` | early exit on the change tolerance, compliance stability, or functional convergence | relative range `< 0.005` |

Three further metrics are recorded for information only and never gate: `thin_member_fraction` (solid
material in members narrower than the filter length scale, measured by a morphological opening),
`checkerboard_index` (mean strength of the alternating pattern over $2\times2$ blocks) and
`load_path_efficiency` (share of the strain energy carried by the connected load path).

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

## Versions
- **v0** -- the original density-filter optimizer with an optimality criteria update.
- **v1** -- identical to v0 apart from a fix to the warm-start path, which adds a small epsilon to the
  provided design to avoid zero-density gradient issues. The datasets are unchanged.
- **v2** -- the AutoSiMP three-field solver, continuation controllers, structural evaluator and
  closed-loop retry described above. `simulate` is unchanged, so the v0 datasets remain valid and
  objective values stay comparable; only `optimize` behaves differently. On the default $100\times50$
  problem it reaches both a markedly lower compliance and a near-binary design where v0/v1 leave a
  substantially gray field.

## Citation
This problem is directly refactored from the [TopOpt-MMA-Python Library](https://github.com/arjendeetman/TopOpt-MMA-Python) and if you use this problem in your experiments, you can use the citations below referencing both the original problem formulation and the subsequent well-known implementation:
```
@article{sigmund200199,
  title={A 99 line topology optimization code written in Matlab},
  author={Sigmund, Ole},
  journal={Structural and multidisciplinary optimization},
  volume={21},
  number={2},
  pages={120--127},
  year={2001},
  publisher={Springer}
}

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

The v2 optimizer additionally follows the three-field solver, the structural evaluator and the
continuation control of AutoSiMP, whose projection scheme is the one of Wang et al. (2011):
```
@article{yang2026autosimp,
  title={AutoSiMP: Autonomous Topology Optimization from Natural Language via LLM-Driven Problem Configuration and Adaptive Solver Control},
  author={Yang, Shaoliang and Wang, Jun and Wang, Yunsheng},
  journal={arXiv preprint arXiv:2603.27000},
  year={2026}
}

@article{wang2011projection,
  title={On projection methods, convergence and robust formulations in topology optimization},
  author={Wang, Fengwen and Lazarov, Boyan Stefanov and Sigmund, Ole},
  journal={Structural and Multidisciplinary Optimization},
  volume={43},
  number={6},
  pages={767--784},
  year={2011},
  publisher={Springer}
}
```
