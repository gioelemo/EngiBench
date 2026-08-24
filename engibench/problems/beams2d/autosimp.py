# Disabled variable name conventions to stay consistent with the topology optimization literature.

r"""Solver-side building blocks of the AutoSiMP pipeline for the Beams2D problem.

This module implements the parts of AutoSiMP (`arXiv:2603.27000
<https://arxiv.org/abs/2603.27000>`_, *AutoSiMP: Autonomous Topology Optimization
from Natural Language via LLM-Driven Problem Configuration and Adaptive Solver
Control*) that are solver-side and therefore transferable to EngiBench:

* a **three-field SIMP** parameterization (design field :math:`x` → filtered
  field :math:`\tilde{x}` → projected physical field :math:`\bar{x}`) built on a
  smooth Heaviside projection,
* **pluggable continuation control**: a controller is queried at every iteration
  and returns the four *Direct Numeric Control* (DNC) parameters -- penalization
  exponent :math:`p`, projection sharpness :math:`\beta`, filter radius
  :math:`r_{\min}` and OC move limit :math:`\delta`,
* the **structural quality evaluator**: five pass/fail gates (connectivity,
  compliance ratio, grayness, volume fraction, convergence) plus three
  informational metrics (thin-member fraction, checkerboard index, load-path
  efficiency).

The two remaining AutoSiMP modules -- the LLM configurator that parses a plain
English prompt and the boundary-condition generator that turns the resulting
specification into solver arrays -- are intentionally *not* reimplemented here:
EngiBench problems already expose a structured, validated ``Conditions``
dataclass, which is exactly the artifact the configurator is meant to produce,
and EngiBench does not depend on an LLM provider. The
:class:`Controller` protocol below is the extension point an LLM agent plugs
into, so the adaptive-control experiments of the paper can be reproduced
without EngiBench itself calling a model.

.. note::
   The paper text was not reachable from this environment (``arxiv.org`` is
   blocked by the network egress proxy), so the formulas for the individual
   quality metrics follow the standard definitions from the topology
   optimization literature that the paper builds on (Wang et al. 2011 for the
   projection and the grayness measure :math:`M_{nd}`, morphological opening for
   the length-scale check). Thresholds are the ones reported for AutoSiMP.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import ndimage

if TYPE_CHECKING:
    from engibench.problems.beams2d.backend import State

__all__ = [
    "CheckResult",
    "ControlSignal",
    "Controller",
    "ControllerLike",
    "FixedController",
    "Observation",
    "QualityReport",
    "ScheduleController",
    "checkerboard_index",
    "density_filter",
    "evaluate",
    "filter_sensitivity",
    "grayness",
    "heaviside_derivative",
    "heaviside_projection",
    "load_elements",
    "load_path_efficiency",
    "load_path_mask",
    "support_elements",
    "thin_member_fraction",
]

# --------------------------------------------------------------------------------------
# Three-field SIMP: density filter + smooth Heaviside projection
# --------------------------------------------------------------------------------------

#: Below this sharpness the projection is numerically indistinguishable from the identity.
_MIN_BETA = 1e-6


def density_filter(st: State, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Apply the density filter :math:`\tilde{x} = Hx / H_s` to a flat design field.

    Args:
        st: State holding the assembled filter matrix ``H`` and its row sums ``Hs``.
        x: Flat design field of length ``nelx * nely``.

    Returns:
        npt.NDArray: The filtered field, flat and of the same length as ``x``.
    """
    return np.asarray(st.H * np.asarray(x, dtype=float)[np.newaxis].T / st.Hs)[:, 0]


def filter_sensitivity(st: State, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Chain a sensitivity w.r.t. the filtered field back to the design field.

    The density filter of :func:`density_filter` is linear with matrix
    :math:`H / H_s`, so its adjoint is :math:`v \mapsto H (v / H_s)` -- note that
    this is *not* the same expression as the forward filter, even though ``H`` is
    symmetric, because the normalization sits on the other side.

    Args:
        st: State holding the assembled filter matrix ``H`` and its row sums ``Hs``.
        v: Flat sensitivity w.r.t. the filtered field.

    Returns:
        npt.NDArray: The sensitivity w.r.t. the design field.
    """
    return np.asarray(st.H * (np.asarray(v, dtype=float)[np.newaxis].T / st.Hs))[:, 0]


def heaviside_projection(x_tilde: npt.NDArray[np.float64], beta: float, eta: float = 0.5) -> npt.NDArray[np.float64]:
    r"""Smooth Heaviside projection of the filtered field.

    Implements the ``tanh`` projection of Wang, Lazarov and Sigmund (2011),

    .. math::
       \bar{x} = \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde{x} - \eta))}
                      {\tanh(\beta\eta) + \tanh(\beta(1 - \eta))},

    which maps :math:`[0, 1]` onto :math:`[0, 1]` and converges to the exact
    Heaviside step at threshold ``eta`` as ``beta`` grows.

    Args:
        x_tilde: The filtered density field.
        beta: Projection sharpness (``beta -> 0`` recovers the identity).
        eta: Projection threshold, ``0.5`` in AutoSiMP.

    Returns:
        npt.NDArray: The projected (physical) density field.
    """
    if beta <= _MIN_BETA:
        return np.asarray(x_tilde, dtype=float)
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    projected = (np.tanh(beta * eta) + np.tanh(beta * (np.asarray(x_tilde, dtype=float) - eta))) / denominator
    # The map is exactly onto [0, 1]; clip away the rounding noise of the normalization.
    return np.clip(projected, 0.0, 1.0)


def heaviside_derivative(x_tilde: npt.NDArray[np.float64], beta: float, eta: float = 0.5) -> npt.NDArray[np.float64]:
    r"""Derivative :math:`\partial\bar{x}/\partial\tilde{x}` of :func:`heaviside_projection`.

    Args:
        x_tilde: The filtered density field.
        beta: Projection sharpness.
        eta: Projection threshold.

    Returns:
        npt.NDArray: The element-wise derivative of the projection, always positive.
    """
    if beta <= _MIN_BETA:
        return np.ones_like(np.asarray(x_tilde, dtype=float))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (np.asarray(x_tilde, dtype=float) - eta)) ** 2) / denominator


# --------------------------------------------------------------------------------------
# Pluggable continuation control (the "Direct Numeric Control" interface)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlSignal:
    r"""The four solver parameters a continuation controller sets at every iteration.

    Attributes:
        penal: SIMP penalization exponent :math:`p`.
        beta: Heaviside projection sharpness :math:`\beta`.
        rmin: Density filter radius :math:`r_{\min}` (in elements).
        move: Optimality-criteria move limit :math:`\delta`.
    """

    penal: float
    beta: float
    rmin: float
    move: float


@dataclass(frozen=True)
class Observation:
    """The solver state a controller sees before choosing the next :class:`ControlSignal`.

    The fields mirror the structured observation AutoSiMP hands to its adaptive
    controller: the objective and its history, the discreteness of the current
    field, how much of the iteration budget has been consumed and how long the
    design has been stagnating.

    Attributes:
        iteration: Zero-based index of the iteration that is about to run.
        max_iter: Total iteration budget.
        compliance: Compliance of the current design.
        best_compliance: Best (lowest) compliance seen so far in this run.
        grayness: Non-discreteness measure :math:`M_{nd}` of the current field.
        volume_fraction: Mean physical density of the current field.
        checkerboard: Checkerboard index of the current field.
        change: Infinity norm of the last design-variable update.
        stagnation: Number of consecutive iterations with ``change`` below the tolerance.
        budget_used: ``iteration / max_iter``, in ``[0, 1]``.
        signal: The control signal used for the previous iteration.
    """

    iteration: int
    max_iter: int
    compliance: float
    best_compliance: float
    grayness: float
    volume_fraction: float
    checkerboard: float
    change: float
    stagnation: int
    budget_used: float
    signal: ControlSignal


class Controller(Protocol):
    """Direct Numeric Control interface: map an :class:`Observation` to a :class:`ControlSignal`.

    Any callable with this signature can be handed to
    ``Beams2D.optimize(controller=...)``. AutoSiMP compares a fixed baseline, a
    deterministic schedule and an LLM agent through exactly this interface; the
    first two are provided as :class:`FixedController` and
    :class:`ScheduleController`, an LLM agent is supplied by the user.
    """

    def __call__(self, observation: Observation) -> ControlSignal:
        """Return the solver parameters to use for ``observation.iteration``."""
        ...


@dataclass
class FixedController:
    """Baseline controller: hold all four parameters constant for the whole run.

    Attributes:
        penal: Constant penalization exponent.
        beta: Constant projection sharpness.
        rmin: Constant filter radius.
        move: Constant move limit.
    """

    penal: float
    beta: float
    rmin: float
    move: float

    def __call__(self, observation: Observation) -> ControlSignal:
        """Return the constant control signal, ignoring the observation."""
        del observation
        return ControlSignal(penal=self.penal, beta=self.beta, rmin=self.rmin, move=self.move)


@dataclass
class ScheduleController:
    """Deterministic continuation schedule -- AutoSiMP's non-LLM reference controller.

    The schedule is *budget aware*: both continuations are expressed as fractions
    of ``max_iter`` so that a short run still reaches the target penalization and
    sharpness instead of stopping half way through the ramp.

    * **Penalization**: stepped from ``penal_init`` to ``penal_target`` in
      increments of ``penal_step``, spread evenly over the first
      ``penal_fraction`` of the budget.
    * **Sharpness**: doubled from ``beta_init`` up to ``beta_max`` at evenly
      spaced iterations, and doubled early whenever the design stagnates, which
      is the classical trigger of Wang et al. (2011).
    * **Move limit**: decayed linearly from ``move_init`` to ``move_min`` over the
      budget, which damps the late-iteration oscillations the compliance-ratio
      gate is designed to catch.
    * **Filter radius**: held at ``rmin``.

    Attributes:
        penal_target: Final penalization exponent.
        rmin: Filter radius (constant).
        max_iter: Iteration budget the schedule is stretched over.
        penal_init: Initial penalization exponent.
        penal_step: Increment of the penalization continuation.
        penal_fraction: Fraction of the budget over which ``penal`` reaches its target.
        beta_init: Initial projection sharpness.
        beta_max: Maximum projection sharpness.
        move_init: Initial move limit.
        move_min: Final move limit.
        stagnation_trigger: Consecutive stagnating iterations that force a ``beta`` doubling.
    """

    penal_target: float
    rmin: float
    max_iter: int
    penal_init: float = 1.0
    penal_step: float = 0.5
    penal_fraction: float = 0.4
    beta_init: float = 1.0
    beta_max: float = 16.0
    move_init: float = 0.2
    move_min: float = 0.05
    stagnation_trigger: int = 2

    _extra_doublings: int = field(default=0, init=False, repr=False)
    _last_doubling: int = field(default=-1, init=False, repr=False)

    def reset(self) -> None:
        """Forget the stagnation-triggered ``beta`` doublings of a previous run."""
        self._extra_doublings = 0
        self._last_doubling = -1

    @property
    def n_doublings(self) -> int:
        """Number of doublings needed to go from ``beta_init`` to ``beta_max``."""
        if self.beta_max <= self.beta_init:
            return 0
        return math.ceil(math.log2(self.beta_max / self.beta_init))

    def _penal(self, iteration: int) -> float:
        n_steps = max(1, math.ceil((self.penal_target - self.penal_init) / max(self.penal_step, 1e-12)))
        ramp = max(1, round(self.penal_fraction * self.max_iter))
        step = int(iteration * n_steps / ramp)
        return float(min(self.penal_target, self.penal_init + step * self.penal_step))

    def _beta(self, iteration: int) -> float:
        n = self.n_doublings
        scheduled = 0 if n == 0 else int(iteration * (n + 1) / max(1, self.max_iter))
        doublings = min(n, scheduled + self._extra_doublings)
        return float(min(self.beta_max, self.beta_init * 2.0**doublings))

    def _move(self, iteration: int) -> float:
        progress = min(1.0, iteration / max(1, self.max_iter))
        return float(self.move_init + (self.move_min - self.move_init) * progress)

    def __call__(self, observation: Observation) -> ControlSignal:
        """Return the scheduled control signal for ``observation.iteration``."""
        if (
            observation.stagnation >= self.stagnation_trigger
            and observation.iteration > self._last_doubling
            and self._beta(observation.iteration) < self.beta_max
        ):
            self._extra_doublings += 1
            self._last_doubling = observation.iteration
        return ControlSignal(
            penal=self._penal(observation.iteration),
            beta=self._beta(observation.iteration),
            rmin=self.rmin,
            move=self._move(observation.iteration),
        )


# --------------------------------------------------------------------------------------
# Structural quality metrics
# --------------------------------------------------------------------------------------

#: 4-neighbour (von Neumann) connectivity structure used for the flood fill.
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def grayness(x: npt.NDArray[np.float64]) -> float:
    r"""Measure of non-discreteness :math:`M_{nd} = \frac{1}{n}\sum_e 4 x_e (1 - x_e)`.

    Zero for a perfectly black-and-white design, one for an all-gray design.

    Args:
        x: Physical density field (any shape).

    Returns:
        float: The grayness of the field, in ``[0, 1]``.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean(4.0 * x * (1.0 - x)))


def checkerboard_index(x_2d: npt.NDArray[np.float64]) -> float:
    r"""Mean strength of the alternating pattern over all ``2x2`` element blocks.

    For a block :math:`\begin{pmatrix} a & b \\ c & d\end{pmatrix}` the local
    indicator is :math:`|a - b - c + d| / 2`, which is ``1`` for a perfect
    checkerboard and ``0`` for any locally constant or linearly varying field.

    Args:
        x_2d: Physical density field as a 2D array.

    Returns:
        float: The checkerboard index, in ``[0, 1]``.
    """
    x_2d = np.asarray(x_2d, dtype=float)
    if x_2d.shape[0] < 2 or x_2d.shape[1] < 2:  # noqa: PLR2004
        return 0.0
    a = x_2d[:-1, :-1]
    b = x_2d[:-1, 1:]
    c = x_2d[1:, :-1]
    d = x_2d[1:, 1:]
    return float(np.mean(np.abs(a - b - c + d)) / 2.0)


def _disk(radius: float) -> npt.NDArray[np.bool_]:
    """Boolean disk structuring element of the given radius (at least one element wide)."""
    r = max(1, math.floor(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= max(radius, 1.0) ** 2 + 1e-9


def thin_member_fraction(x_2d: npt.NDArray[np.float64], rmin: float, threshold: float = 0.5) -> float:
    """Fraction of the solid material sitting in members thinner than the length scale.

    Computed by a morphological opening of the thresholded design with a disk of
    radius ``rmin / 2``: material that the opening removes belongs to features
    narrower than the filter's length scale.

    Args:
        x_2d: Physical density field as a 2D array.
        rmin: Filter radius, i.e. the requested minimum feature length.
        threshold: Density above which an element counts as solid.

    Returns:
        float: The fraction of solid elements in thin members, in ``[0, 1]``.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    n_solid = int(solid.sum())
    if n_solid == 0:
        return 0.0
    disk = _disk(rmin / 2.0)
    # `border_value=1` keeps material that merely touches the domain boundary from being
    # reported as thin: the boundary is a support, not void.
    eroded = ndimage.binary_erosion(solid, structure=disk, border_value=1)
    opened = ndimage.binary_dilation(eroded, structure=disk)
    return float(np.logical_and(solid, ~opened).sum() / n_solid)


def support_elements(st: State, nelx: int, nely: int) -> npt.NDArray[np.bool_]:
    """Mask of the elements touching a constrained node.

    Args:
        st: State holding the fixed degrees of freedom.
        nelx: Number of elements in the x direction.
        nely: Number of elements in the y direction.

    Returns:
        npt.NDArray: Boolean mask of shape ``(nelx, nely)``.
    """
    return _elements_of_nodes(np.unique(np.asarray(st.fixed, dtype=int) // 2), nelx, nely)


def load_elements(st: State, nelx: int, nely: int) -> npt.NDArray[np.bool_]:
    """Mask of the elements touching a loaded node.

    Args:
        st: State holding the force vector.
        nelx: Number of elements in the x direction.
        nely: Number of elements in the y direction.

    Returns:
        npt.NDArray: Boolean mask of shape ``(nelx, nely)``.
    """
    loaded_dofs = np.flatnonzero(np.asarray(st.f).ravel() != 0.0)
    return _elements_of_nodes(np.unique(loaded_dofs // 2), nelx, nely)


def _elements_of_nodes(nodes: npt.NDArray[np.int_], nelx: int, nely: int) -> npt.NDArray[np.bool_]:
    """Mark every element of the ``(nelx, nely)`` grid that has one of ``nodes`` as a corner."""
    mask = np.zeros((nelx, nely), dtype=bool)
    node_i, node_j = np.divmod(np.asarray(nodes, dtype=int), nely + 1)
    for di in (-1, 0):
        for dj in (-1, 0):
            i = node_i + di
            j = node_j + dj
            valid = (i >= 0) & (i < nelx) & (j >= 0) & (j < nely)
            mask[i[valid], j[valid]] = True
    return mask


def load_path_mask(
    x_2d: npt.NDArray[np.float64],
    supports: npt.NDArray[np.bool_],
    loads: npt.NDArray[np.bool_],
    threshold: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Solid elements of the connected components that link a support to a load.

    A 4-neighbour flood fill is run on the thresholded design; a component is
    part of the load path when it touches both a constrained node and a loaded
    node. An empty mask therefore means the design is disconnected.

    Args:
        x_2d: Physical density field, shape ``(nelx, nely)``.
        supports: Mask of support-adjacent elements, same shape.
        loads: Mask of load-adjacent elements, same shape.
        threshold: Density above which an element counts as solid.

    Returns:
        npt.NDArray: Boolean mask of the load-carrying elements.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    labels, n_labels = ndimage.label(solid, structure=_CROSS)
    if n_labels == 0:
        return np.zeros_like(solid)
    support_labels = set(np.unique(labels[solid & supports]).tolist())
    load_labels = set(np.unique(labels[solid & loads]).tolist())
    common = sorted((support_labels & load_labels) - {0})
    if not common:
        return np.zeros_like(solid)
    return np.isin(labels, common)


def load_path_efficiency(
    x_2d: npt.NDArray[np.float64],
    energy_2d: npt.NDArray[np.float64],
    path: npt.NDArray[np.bool_],
) -> float:
    """Share of the total strain energy carried by the connected load path.

    Values close to one mean that essentially all of the stored energy sits in
    material that actually connects the loads to the supports; a low value
    signals disconnected or parasitic material.

    Args:
        x_2d: Physical density field, shape ``(nelx, nely)`` (unused, kept for symmetry).
        energy_2d: Element-wise strain energy, same shape.
        path: Mask of the load-carrying elements, same shape.

    Returns:
        float: The load-path efficiency, in ``[0, 1]``.
    """
    del x_2d
    total = float(np.sum(energy_2d))
    if total <= 0.0:
        return 0.0
    return float(np.sum(energy_2d[path]) / total)


# --------------------------------------------------------------------------------------
# The eight-check structural evaluator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single evaluator check.

    Attributes:
        name: Identifier of the check.
        value: The measured quantity.
        threshold: The threshold the value is compared against (``None`` for boolean checks).
        passed: Whether the check succeeded. Always ``True`` for informational metrics.
        informational: ``True`` for the three metrics that are recorded but never gate.
        detail: Human readable explanation.
    """

    name: str
    value: float
    threshold: float | None
    passed: bool
    informational: bool = False
    detail: str = ""


@dataclass
class QualityReport:
    """Result of the eight-check structural evaluator.

    Five checks gate the design (``connectivity``, ``compliance_ratio``,
    ``grayness``, ``volume_fraction``, ``convergence``) and three are recorded
    for information only (``thin_member_fraction``, ``checkerboard_index``,
    ``load_path_efficiency``).

    Attributes:
        checks: All eight checks, keyed by name.
        attempt: Zero-based index of the solver attempt this report describes.
    """

    checks: dict[str, CheckResult] = field(default_factory=dict)
    attempt: int = 0

    @property
    def passed(self) -> bool:
        """``True`` when every gating check succeeded."""
        return all(c.passed for c in self.checks.values() if not c.informational)

    @property
    def failed_checks(self) -> list[str]:
        """Names of the gating checks that failed, in evaluation order."""
        return [name for name, c in self.checks.items() if not c.informational and not c.passed]

    @property
    def metrics(self) -> dict[str, float]:
        """The measured value of every check, keyed by name."""
        return {name: c.value for name, c in self.checks.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view of the report, suitable for logging."""
        return {
            "attempt": self.attempt,
            "passed": self.passed,
            "checks": {name: dataclasses.asdict(c) for name, c in self.checks.items()},
        }

    def __str__(self) -> str:
        """Return a one-line-per-check summary of the report."""
        lines = [f"QualityReport(attempt={self.attempt}, passed={self.passed})"]
        for name, c in self.checks.items():
            mark = "info" if c.informational else ("pass" if c.passed else "FAIL")
            lines.append(f"  [{mark}] {name}: {c.value:.4g}" + (f" (threshold {c.threshold:.4g})" if c.threshold else ""))
        return "\n".join(lines)


def evaluate(  # noqa: PLR0913
    x_2d: npt.NDArray[np.float64],
    energy_2d: npt.NDArray[np.float64],
    st: State,
    *,
    volfrac: float,
    rmin: float,
    compliance_history: list[float],
    early_exit: bool,
    attempt: int = 0,
    grayness_tol: float = 0.15,
    volfrac_tol: float = 0.02,
    compliance_ratio_tol: float = 2.0,
    stability_tol: float = 0.005,
    stability_window: int = 10,
    threshold: float = 0.5,
) -> QualityReport:
    """Run the eight-check structural evaluator on a finished optimization.

    Args:
        x_2d: Final physical density field, shape ``(nelx, nely)``.
        energy_2d: Element-wise strain energy of the final design, same shape.
        st: State holding the boundary conditions (used for supports and loads).
        volfrac: Target volume fraction.
        rmin: Filter radius, i.e. the requested minimum feature length.
        compliance_history: Compliance recorded at every iteration.
        early_exit: Whether the solver stopped on its change tolerance rather than
            exhausting the iteration budget.
        attempt: Zero-based index of the solver attempt being evaluated.
        grayness_tol: Maximum admissible :math:`M_{nd}`.
        volfrac_tol: Maximum admissible *relative* deviation from ``volfrac``.
        compliance_ratio_tol: Maximum admissible ``final / best`` compliance ratio.
        stability_tol: Maximum admissible relative range of the last compliances.
        stability_window: Number of trailing iterations used for the stability test.
        threshold: Density above which an element counts as solid.

    Returns:
        QualityReport: The five gating checks and the three informational metrics.
    """
    nelx, nely = x_2d.shape
    supports = support_elements(st, nelx, nely)
    loads = load_elements(st, nelx, nely)
    path = load_path_mask(x_2d, supports, loads, threshold=threshold)
    connected = bool(path.any())

    history = [float(c) for c in compliance_history]
    final_c = history[-1] if history else math.inf
    best_c = min(history) if history else math.inf
    ratio = final_c / best_c if best_c > 0 and math.isfinite(best_c) else math.inf

    gray = grayness(x_2d)
    actual_volfrac = float(np.mean(x_2d))
    volfrac_dev = abs(actual_volfrac - volfrac) / volfrac if volfrac > 0 else math.inf

    window = history[-stability_window:]
    if len(window) >= 2:  # noqa: PLR2004
        mean_c = float(np.mean(window))
        stability = float(np.ptp(window) / mean_c) if mean_c > 0 else math.inf
    else:
        stability = math.inf
    # AutoSiMP accepts a run as converged on an early exit, on compliance stability, or on
    # "functional" convergence: a design that is already discrete and volume-feasible and
    # whose objective has stopped improving does not need the rest of its budget.
    functional = gray <= grayness_tol and volfrac_dev <= volfrac_tol and stability <= 10.0 * stability_tol
    converged = early_exit or stability < stability_tol or functional

    energy = np.asarray(energy_2d, dtype=float)
    checks = {
        "connectivity": CheckResult(
            name="connectivity",
            value=float(connected),
            threshold=None,
            passed=connected,
            detail="4-neighbour flood fill links a loaded node to a constrained node",
        ),
        "compliance_ratio": CheckResult(
            name="compliance_ratio",
            value=ratio,
            threshold=compliance_ratio_tol,
            passed=ratio < compliance_ratio_tol,
            detail="final compliance relative to the best compliance seen (tail degradation)",
        ),
        "grayness": CheckResult(
            name="grayness",
            value=gray,
            threshold=grayness_tol,
            passed=gray <= grayness_tol,
            detail="non-discreteness measure M_nd of the final density field",
        ),
        "volume_fraction": CheckResult(
            name="volume_fraction",
            value=actual_volfrac,
            threshold=volfrac_tol,
            passed=volfrac_dev <= volfrac_tol,
            detail=f"relative deviation {volfrac_dev:.4f} from the target {volfrac:.4f}",
        ),
        "convergence": CheckResult(
            name="convergence",
            value=stability,
            threshold=stability_tol,
            passed=converged,
            detail=f"early_exit={early_exit}, stability={stability:.4g}, functional={functional}",
        ),
        "thin_member_fraction": CheckResult(
            name="thin_member_fraction",
            value=thin_member_fraction(x_2d, rmin, threshold=threshold),
            threshold=None,
            passed=True,
            informational=True,
            detail="solid material in members narrower than the filter length scale",
        ),
        "checkerboard_index": CheckResult(
            name="checkerboard_index",
            value=checkerboard_index(x_2d),
            threshold=None,
            passed=True,
            informational=True,
            detail="mean strength of the alternating pattern over 2x2 blocks",
        ),
        "load_path_efficiency": CheckResult(
            name="load_path_efficiency",
            value=load_path_efficiency(x_2d, energy, path),
            threshold=None,
            passed=True,
            informational=True,
            detail="share of the strain energy carried by the connected load path",
        ),
    }
    return QualityReport(checks=checks, attempt=attempt)


#: Convenience alias for anything usable as a continuation controller.
ControllerLike = Controller | Callable[[Observation], ControlSignal]
