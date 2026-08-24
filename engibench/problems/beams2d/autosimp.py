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
   Where AutoSiMP's absolute solver values collide with an EngiBench *condition*
   -- the penalization exponent ``penal`` and the filter radius ``rmin`` are part
   of this problem's definition and of its datasets -- the condition wins and the
   paper's value is offered as an opt-in default. This affects the sharpening
   tail only; see :class:`TailSpec`.
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
    "TailSpec",
    "ThreeFieldController",
    "checkerboard_index",
    "connectivity_fraction",
    "density_filter",
    "evaluate",
    "filter_sensitivity",
    "grayness",
    "heaviside_derivative",
    "heaviside_projection",
    "load_elements",
    "load_path_efficiency",
    "reachable_from_supports",
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
    ``Beams2D.optimize(controller=...)``. AutoSiMP compares six controllers -- an
    LLM agent, a deterministic schedule, an expert heuristic, the standard
    three-field continuation, a tail-only ablation and a fixed baseline --
    through exactly this interface; :class:`ScheduleController`,
    :class:`ThreeFieldController` and :class:`FixedController` are provided here,
    an LLM agent is supplied by the user.

    Three further methods are optional and are used when present:

    * ``initialize() -> ControlSignal`` -- the starting parameters, used before the
      first observation exists. Defaults to the configured initial values.
    * ``finalize() -> TailSpec | None`` -- the sharpening tail to run after the main
      loop, restarting from the best valid snapshot. ``None`` means no tail.
    * ``reset() -> None`` -- clear any state carried over from a previous attempt.
    """

    def __call__(self, observation: Observation) -> ControlSignal:
        """Return the solver parameters to use for ``observation.iteration``."""
        ...


@dataclass(frozen=True)
class TailSpec:
    r"""The sharpening tail that every non-fixed AutoSiMP controller shares.

    The paper runs an identical 40-iteration tail at :math:`p = 4.5`,
    :math:`\beta = 32`, :math:`r_{\min} = 1.20` and :math:`\delta = 0.05`,
    restarting from the best snapshot that satisfied the validity gate, so that
    compliance differences between controllers are attributable to the
    exploration phase alone.

    In EngiBench ``penal`` and ``rmin`` are *conditions*: they are part of the
    problem definition and of the datasets keyed on it. The tail therefore keeps
    whatever the conditions ask for by default, and the paper's ``4.5`` / ``1.20``
    are reached by setting ``tail_penal`` and ``tail_rmin`` explicitly. ``beta``
    and ``move`` are pure solver knobs and take the paper's values.

    Attributes:
        penal: Penalization exponent held for the whole tail.
        rmin: Filter radius held for the whole tail.
        iterations: Length of the tail.
        beta: Projection sharpness held for the whole tail.
        move: Optimality-criteria move limit held for the whole tail.
    """

    penal: float
    rmin: float
    iterations: int = 40
    beta: float = 32.0
    move: float = 0.05

    def signal(self) -> ControlSignal:
        """Return the constant control signal of the tail."""
        return ControlSignal(penal=self.penal, beta=self.beta, rmin=self.rmin, move=self.move)


@dataclass
class FixedController:
    r"""Baseline controller: hold all four parameters constant and run no tail.

    This is AutoSiMP's true no-intervention baseline, which the paper runs at
    :math:`p = 3` and :math:`\beta = 1` and which produces gray, unconverged
    designs.

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

    def initialize(self) -> ControlSignal:
        """Return the constant control signal."""
        return ControlSignal(penal=self.penal, beta=self.beta, rmin=self.rmin, move=self.move)

    def finalize(self) -> TailSpec | None:
        """Return ``None``: the fixed baseline deliberately runs no sharpening tail."""
        return None


@dataclass
class ScheduleController:
    """AutoSiMP's deterministic schedule -- the recommended default controller.

    It reproduces the four-stage phase structure of the LLM agent with
    pre-computed values, which is the paper's key ablation: it isolates the
    contribution of the phase structure from the LLM's adaptive decisions, and
    reaches a 100% pass rate at only +1.5% median compliance over the LLM agent.

    The phases are expressed as fractions of the exploration budget, so a short
    run still traverses all four instead of stopping half way through:

    * **exploration** (to ``exploration_end``): low penalization and a mild
      projection, to discover a topology rather than commit to one.
    * **penalization** (to ``penalization_end``): ``penal`` stepped up to its
      target in ``penal_step`` increments.
    * **sharpening** (to ``sharpening_end``): ``beta`` doubled geometrically up to
      ``beta_max``.
    * **convergence** (the remainder): both held at their targets and the move
      limit dropped to ``move_min`` to settle the design.

    ``beta`` is additionally doubled early whenever the design stagnates, the
    classical trigger of Wang et al. (2011). The sharpening tail returned by
    :meth:`finalize` then takes over.

    .. note::
       The paper names the four phases but takes their numeric values from its
       companion controller paper (arXiv:2603.25099), which it does not reproduce.
       The default fractions here were chosen on a grid of Beams2D conditions and
       validated on a disjoint one, not read off the paper.
       :class:`ThreeFieldController` implements the one baseline whose values the
       paper does state in full.

    Attributes:
        penal_target: Final penalization exponent.
        rmin: Filter radius (constant; the tail may change it).
        max_iter: Exploration budget the phases are stretched over.
        tail: The sharpening tail, or ``None`` to run none.
        penal_init: Initial penalization exponent.
        penal_step: Increment of the penalization continuation.
        beta_init: Initial projection sharpness.
        beta_max: Maximum projection sharpness of the exploration phase.
        move_init: Initial move limit.
        move_min: Move limit of the convergence phase.
        exploration_end: Fraction of the budget at which the exploration phase ends.
        penalization_end: Fraction of the budget at which the penalization phase ends.
        sharpening_end: Fraction of the budget at which the sharpening phase ends.
        stagnation_trigger: Consecutive stagnating iterations that force a ``beta`` doubling.
    """

    penal_target: float
    rmin: float
    max_iter: int
    tail: TailSpec | None = None
    penal_init: float = 1.0
    penal_step: float = 0.5
    beta_init: float = 1.0
    beta_max: float = 16.0
    move_init: float = 0.2
    move_min: float = 0.05
    exploration_end: float = 0.15
    penalization_end: float = 0.55
    sharpening_end: float = 0.85
    stagnation_trigger: int = 2

    _extra_doublings: int = field(default=0, init=False, repr=False)
    _last_doubling: int = field(default=-1, init=False, repr=False)

    def reset(self) -> None:
        """Forget the stagnation-triggered ``beta`` doublings of a previous run."""
        self._extra_doublings = 0
        self._last_doubling = -1

    def initialize(self) -> ControlSignal:
        """Return the parameters of the first iteration."""
        return ControlSignal(penal=self.penal_init, beta=self.beta_init, rmin=self.rmin, move=self.move_init)

    def finalize(self) -> TailSpec | None:
        """Return the sharpening tail to run after the exploration budget."""
        return self.tail

    @property
    def n_doublings(self) -> int:
        """Number of doublings needed to go from ``beta_init`` to ``beta_max``."""
        if self.beta_max <= self.beta_init:
            return 0
        return math.ceil(math.log2(self.beta_max / self.beta_init))

    def _progress(self, iteration: int) -> float:
        return min(1.0, iteration / max(1, self.max_iter))

    def _penal(self, iteration: int) -> float:
        progress = self._progress(iteration)
        if progress < self.exploration_end:
            return float(self.penal_init)
        span = max(self.penalization_end - self.exploration_end, 1e-12)
        ramp = min(1.0, (progress - self.exploration_end) / span)
        n_steps = max(1, math.ceil((self.penal_target - self.penal_init) / max(self.penal_step, 1e-12)))
        step = min(n_steps, int(ramp * n_steps))
        return float(min(self.penal_target, self.penal_init + step * self.penal_step))

    def _beta(self, iteration: int) -> float:
        n = self.n_doublings
        progress = self._progress(iteration)
        if progress < self.penalization_end:
            scheduled = 0
        elif progress >= self.sharpening_end:
            scheduled = n
        else:
            span = max(self.sharpening_end - self.penalization_end, 1e-12)
            scheduled = min(n, int((progress - self.penalization_end) / span * (n + 1)))
        doublings = min(n, scheduled + self._extra_doublings)
        return float(min(self.beta_max, self.beta_init * 2.0**doublings))

    def _move(self, iteration: int) -> float:
        if self._progress(iteration) >= self.sharpening_end:
            return float(self.move_min)
        return float(self.move_init)

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


@dataclass
class ThreeFieldController:
    """The standard academic three-field continuation, AutoSiMP's baseline 4.

    The paper states this baseline in full: a linear ``penal`` ramp from
    ``penal_init`` to ``penal_target`` over ``penal_ramp`` iterations, geometric
    ``beta`` doubling every ``beta_interval`` iterations, and a late tightening of
    the filter radius. It follows Wang et al. (2011) and Lazarov et al. (2016).

    ``rmin_final`` defaults to ``rmin``, i.e. no tightening, because ``rmin`` is an
    EngiBench condition; set it explicitly to reproduce the paper's late
    tightening.

    Attributes:
        penal_target: Final penalization exponent.
        rmin: Filter radius before the late tightening.
        max_iter: Exploration budget.
        tail: The sharpening tail, or ``None`` to run none.
        penal_init: Initial penalization exponent.
        penal_ramp: Number of iterations the linear ``penal`` ramp spans.
        beta_init: Initial projection sharpness.
        beta_max: Maximum projection sharpness.
        beta_interval: Iterations between two ``beta`` doublings.
        move: Constant move limit.
        rmin_final: Filter radius after the late tightening.
        tighten_at: Fraction of the budget at which the radius is tightened.
    """

    penal_target: float
    rmin: float
    max_iter: int
    tail: TailSpec | None = None
    penal_init: float = 1.0
    penal_ramp: int = 30
    beta_init: float = 1.0
    beta_max: float = 16.0
    beta_interval: int = 10
    move: float = 0.2
    rmin_final: float | None = None
    tighten_at: float = 0.75

    def initialize(self) -> ControlSignal:
        """Return the parameters of the first iteration."""
        return ControlSignal(penal=self.penal_init, beta=self.beta_init, rmin=self.rmin, move=self.move)

    def finalize(self) -> TailSpec | None:
        """Return the sharpening tail to run after the exploration budget."""
        return self.tail

    def __call__(self, observation: Observation) -> ControlSignal:
        """Return the continuation signal for ``observation.iteration``."""
        k = observation.iteration
        ramp = min(1.0, k / max(1, self.penal_ramp))
        penal = self.penal_init + ramp * (self.penal_target - self.penal_init)
        beta = min(self.beta_max, self.beta_init * 2.0 ** (k // max(1, self.beta_interval)))
        late = k >= self.tighten_at * max(1, self.max_iter)
        rmin = self.rmin_final if (late and self.rmin_final is not None) else self.rmin
        return ControlSignal(penal=float(penal), beta=float(beta), rmin=float(rmin), move=self.move)


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


def thin_member_fraction(x_2d: npt.NDArray[np.float64], threshold: float = 0.5) -> float:
    """Fraction of the solid material sitting in one-element-wide connections (metric 6).

    An element counts as one-element-wide when both of its neighbours along at
    least one axis are void, the domain boundary counting as void. Such elements
    are the members a mesh refinement would not resolve.

    Args:
        x_2d: Physical density field as a 2D array.
        threshold: Density above which an element counts as solid.

    Returns:
        float: The fraction of solid elements in one-element-wide members, in ``[0, 1]``.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    n_solid = int(solid.sum())
    if n_solid == 0:
        return 0.0
    padded = np.pad(solid, 1, constant_values=False)
    left, right = padded[:-2, 1:-1], padded[2:, 1:-1]
    up, down = padded[1:-1, :-2], padded[1:-1, 2:]
    thin = solid & ((~left & ~right) | (~up & ~down))
    return float(thin.sum() / n_solid)


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


def reachable_from_supports(
    x_2d: npt.NDArray[np.float64],
    supports: npt.NDArray[np.bool_],
    threshold: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Solid elements a flood fill from the supports can reach.

    Args:
        x_2d: Physical density field, shape ``(nelx, nely)``.
        supports: Mask of support-adjacent elements, same shape.
        threshold: Density above which an element counts as solid.

    Returns:
        npt.NDArray: Boolean mask of the reached solid elements.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    labels, n_labels = ndimage.label(solid, structure=_CROSS)
    if n_labels == 0:
        return np.zeros_like(solid)
    seeded = sorted(set(np.unique(labels[solid & supports]).tolist()) - {0})
    if not seeded:
        return np.zeros_like(solid)
    return np.isin(labels, seeded) & solid


def connectivity_fraction(
    x_2d: npt.NDArray[np.float64],
    supports: npt.NDArray[np.bool_],
    threshold: float = 0.5,
) -> float:
    r"""Share of the solid material a flood fill from the supports reaches (gate 1).

    .. math::
       f_{\mathrm{conn}} = \frac{|\{e : \tilde{\rho}_e > 0.5 \text{ and } e \text{ reached}\}|}
                                {|\{e : \tilde{\rho}_e > 0.5\}|}

    The flood fill uses 4-connectivity. A value below one means part of the solid
    material floats free of the supports.

    Args:
        x_2d: Physical density field, shape ``(nelx, nely)``.
        supports: Mask of support-adjacent elements, same shape.
        threshold: Density above which an element counts as solid.

    Returns:
        float: The connectivity fraction, in ``[0, 1]``; ``0`` for an all-void design.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    n_solid = int(solid.sum())
    if n_solid == 0:
        return 0.0
    return float(reachable_from_supports(x_2d, supports, threshold).sum() / n_solid)


def _bfs_distances(solid: npt.NDArray[np.bool_], seeds: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """4-connected BFS distance, in element steps, from ``seeds`` through ``solid``."""
    dist = np.full(solid.shape, np.inf)
    frontier = list(zip(*np.nonzero(seeds & solid), strict=True))
    for i, j in frontier:
        dist[i, j] = 0.0
    nelx, nely = solid.shape
    step = 0.0
    while frontier:
        step += 1.0
        nxt = []
        for i, j in frontier:
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = i + di, j + dj
                if 0 <= a < nelx and 0 <= b < nely and solid[a, b] and dist[a, b] > step:
                    dist[a, b] = step
                    nxt.append((a, b))
        frontier = nxt
    return dist


def load_path_efficiency(
    x_2d: npt.NDArray[np.float64],
    supports: npt.NDArray[np.bool_],
    loads: npt.NDArray[np.bool_],
    threshold: float = 0.5,
) -> float:
    """Detour factor of the load path: BFS path length over Euclidean distance (metric 8).

    A 4-connected BFS runs from the loaded elements through solid material to the
    nearest reachable support element; the number of steps it needs is divided by
    the straight-line distance between those two elements. The ideal value is
    ``1.0`` (a straight load path); larger values mean the force detours. Returns
    infinity when no support is reachable from the load.

    Args:
        x_2d: Physical density field, shape ``(nelx, nely)``.
        supports: Mask of support-adjacent elements, same shape.
        loads: Mask of load-adjacent elements, same shape.
        threshold: Density above which an element counts as solid.

    Returns:
        float: The ratio of the BFS path length to the Euclidean distance.
    """
    solid = np.asarray(x_2d, dtype=float) >= threshold
    if not (solid & loads).any() or not (solid & supports).any():
        return math.inf
    dist = _bfs_distances(solid, loads)
    reachable = solid & supports & np.isfinite(dist)
    if not reachable.any():
        return math.inf
    candidates = np.array(np.nonzero(reachable)).T
    target = candidates[int(np.argmin(dist[reachable]))]
    path = float(dist[target[0], target[1]])
    sources = np.array(np.nonzero(solid & loads)).T
    euclidean = float(np.min(np.linalg.norm(sources - target, axis=1)))
    if euclidean <= 0.0:
        return 1.0
    return path / euclidean


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
    st: State,
    *,
    volfrac: float,
    compliance_history: list[float],
    early_exit: bool,
    attempt: int = 0,
    connectivity_tol: float = 0.99,
    grayness_tol: float = 0.15,
    volfrac_tol: float = 0.02,
    compliance_ratio_tol: float = 2.0,
    stability_tol: float = 0.005,
    stability_window: int = 10,
    threshold: float = 0.5,
) -> QualityReport:
    """Run the eight-check structural evaluator of AutoSiMP (its Table 2) on a finished solve.

    Args:
        x_2d: Final physical density field, shape ``(nelx, nely)``.
        st: State holding the boundary conditions (used for supports and loads).
        volfrac: Target volume fraction.
        compliance_history: Compliance recorded at every iteration.
        early_exit: Whether the solver stopped on its change tolerance rather than
            exhausting the iteration budget.
        attempt: Zero-based index of the solver attempt being evaluated.
        connectivity_tol: Minimum admissible connectivity fraction.
        grayness_tol: Maximum admissible grayness index :math:`G`.
        volfrac_tol: Maximum admissible *absolute* deviation from ``volfrac``.
        compliance_ratio_tol: Maximum admissible ``final / best`` compliance ratio. Note that the
            best compliance is normally reached early, while the penalization is still low and the
            field still gray, so this gate charges the cost of binarizing as well as any genuine
            degradation in the tail; raise it to isolate the latter.
        stability_tol: Maximum admissible relative range of the last compliances.
        stability_window: Number of trailing iterations used for the stability test.
        threshold: Density above which an element counts as solid.

    Returns:
        QualityReport: The five gating checks and the three informational metrics.
    """
    nelx, nely = x_2d.shape
    supports = support_elements(st, nelx, nely)
    loads = load_elements(st, nelx, nely)
    f_conn = connectivity_fraction(x_2d, supports, threshold=threshold)

    history = [float(c) for c in compliance_history]
    final_c = history[-1] if history else math.inf
    best_c = min(history) if history else math.inf
    ratio = final_c / best_c if best_c > 0 and math.isfinite(best_c) else math.inf

    gray = grayness(x_2d)
    actual_volfrac = float(np.mean(x_2d))
    volfrac_dev = abs(actual_volfrac - volfrac)

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

    checks = {
        "connectivity": CheckResult(
            name="connectivity",
            value=f_conn,
            threshold=connectivity_tol,
            passed=f_conn >= connectivity_tol,
            detail="share of the solid material reached by a 4-neighbour flood fill from the supports",
        ),
        "compliance_ratio": CheckResult(
            name="compliance_ratio",
            value=ratio,
            threshold=compliance_ratio_tol,
            passed=ratio < compliance_ratio_tol,
            detail="final compliance relative to the best compliance seen over the whole run",
        ),
        "grayness": CheckResult(
            name="grayness",
            value=gray,
            threshold=grayness_tol,
            passed=gray <= grayness_tol,
            detail="grayness index G of the final density field",
        ),
        "volume_fraction": CheckResult(
            name="volume_fraction",
            value=actual_volfrac,
            threshold=volfrac_tol,
            passed=volfrac_dev <= volfrac_tol,
            detail=f"absolute deviation {volfrac_dev:.4f} from the target {volfrac:.4f}",
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
            value=thin_member_fraction(x_2d, threshold=threshold),
            threshold=None,
            passed=True,
            informational=True,
            detail="solid material in one-element-wide connections",
        ),
        "checkerboard_index": CheckResult(
            name="checkerboard_index",
            value=checkerboard_index(x_2d),
            threshold=None,
            passed=True,
            informational=True,
            detail="diagonal contrast over 2x2 element blocks",
        ),
        "load_path_efficiency": CheckResult(
            name="load_path_efficiency",
            value=load_path_efficiency(x_2d, supports, loads, threshold=threshold),
            threshold=None,
            passed=True,
            informational=True,
            detail="BFS path length from the load to the nearest support over their Euclidean distance",
        ),
    }
    return QualityReport(checks=checks, attempt=attempt)


#: Convenience alias for anything usable as a continuation controller.
ControllerLike = Controller | Callable[[Observation], ControlSignal]
