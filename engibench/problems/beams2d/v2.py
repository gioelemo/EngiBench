# ruff: noqa: N806
# Disabled variable name conventions

"""Beams 2D problem - AutoSiMP three-field formulation."""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import greater_than
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.problems.beams2d.autosimp import checkerboard_index
from engibench.problems.beams2d.autosimp import ControllerLike
from engibench.problems.beams2d.autosimp import ControlSignal
from engibench.problems.beams2d.autosimp import density_filter
from engibench.problems.beams2d.autosimp import evaluate
from engibench.problems.beams2d.autosimp import filter_sensitivity
from engibench.problems.beams2d.autosimp import FixedController
from engibench.problems.beams2d.autosimp import grayness
from engibench.problems.beams2d.autosimp import heaviside_derivative
from engibench.problems.beams2d.autosimp import heaviside_projection
from engibench.problems.beams2d.autosimp import Observation
from engibench.problems.beams2d.autosimp import QualityReport
from engibench.problems.beams2d.autosimp import ScheduleController
from engibench.problems.beams2d.autosimp import TailSpec
from engibench.problems.beams2d.autosimp import ThreeFieldController
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import h_mat
from engibench.problems.beams2d.backend import image_to_design
from engibench.problems.beams2d.backend import overhang_filter_d
from engibench.problems.beams2d.backend import overhang_filter_x
from engibench.problems.beams2d.backend import State
from engibench.problems.beams2d.v0 import ExtendedOptiStep
from engibench.problems.beams2d.v0 import main
from engibench.problems.beams2d.v1 import Beams2D as Beams2D_v1
from engibench.utils.upcast import upcast

CONTINUATION_MODES = ("schedule", "three_field", "fixed")


def _validate_continuation(continuation: str) -> None:
    """Raise unless ``continuation`` names one of the built-in controllers."""
    assert continuation in CONTINUATION_MODES, f"Config.continuation: {continuation!r} ∉ {CONTINUATION_MODES}"


@dataclass
class ControlledOptiStep(ExtendedOptiStep):
    """An :class:`ExtendedOptiStep` that also records the solver parameters used for the step.

    Attributes:
        penal: SIMP penalization exponent used for this iteration.
        beta: Heaviside projection sharpness used for this iteration.
        rmin: Density filter radius used for this iteration.
        move: Optimality-criteria move limit used for this iteration.
        attempt: Zero-based index of the solver attempt this step belongs to.
        phase: Solver phase this step belongs to, ``"exploration"`` or ``"tail"``.
    """

    penal: float = 3.0
    beta: float = 1.0
    rmin: float = 2.0
    move: float = 0.2
    attempt: int = 0
    phase: str = "exploration"


class Beams2D(Beams2D_v1):
    r"""Beam 2D topology optimization problem - Version 2 (v2).

    ## v2

    v2 replaces the two-field density-filter optimizer of v0/v1 with the
    **three-field SIMP** solver and the **structural quality evaluator** of
    AutoSiMP ([arXiv:2603.27000](https://arxiv.org/abs/2603.27000)).

    The design field $x$ is filtered into $\tilde{x} = Hx/H_s$ and then pushed
    through a smooth Heaviside projection

    $$\bar{x} = \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde{x}-\eta))}
                     {\tanh(\beta\eta) + \tanh(\beta(1-\eta))},$$

    and it is the projected field $\bar{x}$ that enters the stiffness assembly,
    the volume constraint and the returned design. Sensitivities are chained back
    through the projection and the filter, so the optimizer remains exactly
    consistent with the physics it reports.

    Four further AutoSiMP components are exposed:

    * **Pluggable continuation control.** At every iteration a controller
      receives an [`Observation`][engibench.problems.beams2d.autosimp.Observation]
      and returns the four *Direct Numeric Control* parameters $(p, \beta,
      r_{\min}, \delta)$. `continuation` selects
      [`ScheduleController`][engibench.problems.beams2d.autosimp.ScheduleController]
      (`"schedule"`, the paper's deterministic four-phase schedule and the default),
      [`ThreeFieldController`][engibench.problems.beams2d.autosimp.ThreeFieldController]
      (`"three_field"`, the standard academic continuation the paper uses as a
      baseline) or [`FixedController`][engibench.problems.beams2d.autosimp.FixedController]
      (`"fixed"`, no continuation and no tail). Any callable can be passed through
      `optimize(controller=...)` -- that is the hook an LLM agent plugs into.
    * **Sharpening tail with a validity gate.** Every non-fixed controller ends
      with a short tail at high sharpness and a small move limit, restarted from
      the best snapshot that reached the target penalization while satisfying the
      volume constraint, or from uniform density if there is none. The paper
      reports this tail as the primary driver of final topology quality.
    * **Eight-check structural evaluator.** Connectivity, compliance ratio,
      grayness, volume fraction and convergence gate the run; thin-member
      fraction, checkerboard index and load-path efficiency are recorded for
      information. The report of the last `optimize` call is available as
      `problem.last_quality_report`, and any design can be checked with
      [`evaluate_quality`][engibench.problems.beams2d.v2.Beams2D.evaluate_quality].
    * **Closed-loop retry.** When a gating check fails, the evaluator's rerun hint
      grows the iteration budget by 30% or adjusts the volume target by the
      observed deviation, and the solve is repeated up to `max_retries` times.

    `simulate` is unchanged: the compliance of a given density field is the same
    physics as in v0/v1, so objective values stay comparable across versions and
    the v0 datasets remain valid. Where the paper's absolute solver values collide
    with an EngiBench *condition* -- it runs its tail at $p = 4.5$ and
    $r_{\min} = 1.20$, both of which are part of this problem's definition and of
    its datasets -- the condition wins by default and the paper's value is one
    config entry away (`tail_penal`, `tail_rmin`).

    The remaining two AutoSiMP modules -- the LLM configurator and the
    boundary-condition generator -- are deliberately not reimplemented: the
    validated specification they produce is what EngiBench's `Conditions`
    dataclass already is, and EngiBench does not depend on an LLM provider.
    """

    version = 2

    @dataclass
    class Config(Beams2D_v1.Config):
        """Configuration of the AutoSiMP three-field solver.

        Extends the v0/v1 configuration with the projection, continuation,
        evaluator and retry settings.
        """

        eta: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Heaviside projection threshold"""
        beta_init: Annotated[float, greater_than(0.0).category(THEORY)] = 1.0
        """Initial Heaviside projection sharpness"""
        beta_max: Annotated[
            float, greater_than(0.0).category(THEORY), bounded(lower=1.0, upper=128.0).category(IMPL).warning()
        ] = 16.0
        """Maximum Heaviside projection sharpness reached by the continuation"""
        penal_init: Annotated[float, bounded(lower=1.0).category(IMPL)] = 1.0
        """Initial penalization exponent of the continuation (the target is `penal`)"""
        penal_step: Annotated[float, greater_than(0.0).category(IMPL)] = 0.5
        """Increment of the penalization continuation"""
        penal_gate: float | None = None
        """Penalization above which a snapshot may become the tail's restart point (defaults to `penal`)"""
        move_init: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.2
        """Initial optimality-criteria move limit"""
        move_min: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.05
        """Optimality-criteria move limit of the convergence phase"""
        continuation: str = "schedule"
        """Continuation controller, one of `"schedule"`, `"three_field"` or `"fixed"`"""
        stagnation_patience: Annotated[int, bounded(lower=1).category(IMPL)] = 3
        """Consecutive stagnating iterations required before a phase exits early"""
        tail_iters: Annotated[int, bounded(lower=0).category(IMPL)] = 40
        """Length of the sharpening tail, clamped to half the iteration budget (0 disables it)"""
        tail_beta: Annotated[float, greater_than(0.0).category(IMPL)] = 32.0
        """Projection sharpness held for the whole sharpening tail"""
        tail_penal: float | None = None
        """Penalization of the sharpening tail (defaults to `penal`; the paper uses 4.5)"""
        tail_rmin: float | None = None
        """Filter radius of the sharpening tail (defaults to `rmin`; the paper uses 1.20)"""
        tail_move: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.05
        """Optimality-criteria move limit of the sharpening tail"""
        connectivity_tol: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.99
        """Minimum admissible share of the solid material connected to the supports"""
        grayness_tol: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.15
        """Maximum admissible grayness index of the final design"""
        volfrac_tol: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.02
        """Maximum admissible absolute deviation of the final volume fraction from its target"""
        compliance_ratio_tol: Annotated[float, bounded(lower=1.0).category(IMPL)] = 2.0
        """Maximum admissible ratio between the final and the best compliance"""
        stability_tol: Annotated[float, bounded(lower=0.0).category(IMPL)] = 0.005
        """Maximum admissible relative range of the trailing compliances"""
        stability_window: Annotated[int, bounded(lower=2).category(IMPL)] = 10
        """Number of trailing iterations used by the convergence check"""
        max_retries: Annotated[int, bounded(lower=0).category(THEORY), bounded(upper=10).category(IMPL).warning()] = 2
        """Number of closed-loop retries allowed when the gating checks fail"""

        @constraint
        @staticmethod
        def continuation_mode(continuation: str) -> None:
            """Constraint for continuation ∈ {"schedule", "fixed"}."""
            _validate_continuation(continuation)

    #: The full configuration held by this problem instance, narrowed to the v2 `Config`.
    config: "Beams2D.Config"
    #: Report of the eight-check evaluator for the last `optimize` call.
    last_quality_report: QualityReport | None = None
    #: Reports of every attempt of the last `optimize` call, oldest first.
    quality_reports: tuple[QualityReport, ...] = ()

    def __init__(self, seed: int = 0, config: dict[str, Any] | None = None):
        """Initializes the Beams2D v2 problem.

        Args:
            seed (int): The random seed for the problem.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.
        """
        super().__init__(seed=seed, config=config)
        # `upcast` walks a single step up the MRO, which for the v2 `Config` is the v0 `Config`.
        # Restate the targets explicitly so that `simulate_config` and `conditions` keep their meaning.
        self.simulate_config = upcast(self.config, self.SimulateConfig)
        self.conditions = upcast(self.simulate_config, self.Conditions)
        _validate_continuation(self.config.continuation)
        self.last_quality_report = None
        self.quality_reports = ()

    # ----------------------------------------------------------------------------------
    # Continuation control
    # ----------------------------------------------------------------------------------

    def make_controller(self, config: "Beams2D.Config") -> ControllerLike:
        """Build the continuation controller described by ``config``.

        Args:
            config: The configuration of the run.

        Returns:
            ControllerLike: A :class:`ScheduleController` for ``continuation="schedule"``,
            a :class:`ThreeFieldController` for ``continuation="three_field"`` and a
            :class:`FixedController` for ``continuation="fixed"``.
        """
        _validate_continuation(config.continuation)
        if config.continuation == "fixed":
            # AutoSiMP's no-intervention baseline: no continuation and no sharpening tail.
            return FixedController(penal=config.penal, beta=config.beta_init, rmin=config.rmin, move=config.move_init)

        tail = _make_tail(config)
        if config.continuation == "three_field":
            return ThreeFieldController(
                penal_target=config.penal,
                rmin=config.rmin,
                max_iter=_exploration_budget(config),
                tail=tail,
                penal_init=min(config.penal_init, config.penal),
                beta_init=config.beta_init,
                beta_max=max(config.beta_max, config.beta_init),
                move=config.move_init,
                rmin_final=config.tail_rmin,
            )
        return ScheduleController(
            penal_target=config.penal,
            rmin=config.rmin,
            max_iter=_exploration_budget(config),
            tail=tail,
            penal_init=min(config.penal_init, config.penal),
            penal_step=config.penal_step,
            beta_init=config.beta_init,
            beta_max=max(config.beta_max, config.beta_init),
            move_init=config.move_init,
            move_min=min(config.move_min, config.move_init),
        )

    # ----------------------------------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------------------------------

    def optimize(
        self,
        starting_point: npt.NDArray | None = None,
        config: dict[str, Any] | None = None,
        *,
        controller: ControllerLike | None = None,
    ) -> tuple[np.ndarray, list[ExtendedOptiStep]]:
        """Optimizes the design of a beam with the AutoSiMP three-field solver.

        Each attempt runs an exploration phase under the continuation controller
        followed by the shared sharpening tail, and is then handed to the
        eight-check evaluator. If a gating check fails, the evaluator's rerun hint
        adjusts the settings and the solve is repeated, up to ``max_retries``
        times. As in the paper's Algorithm 1 the loop returns the lowest-compliance
        attempt and stops as soon as one passes; unlike the paper, the report
        stored in :attr:`last_quality_report` is the one of the attempt actually
        returned, so design and report always describe the same beam.

        Args:
            starting_point (npt.NDArray or None): The design to begin warm-start optimization from (optional).
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the optimization.
            controller (ControllerLike or None): A Direct Numeric Control callable mapping an
                :class:`Observation` to a :class:`ControlSignal`, optionally also exposing
                ``initialize()``, ``finalize()`` and ``reset()``. Defaults to the controller
                described by ``config["continuation"]``. An explicit controller is reused as-is
                by every retry, so only the escalations it does not itself override (notably the
                iteration budget) reach it; pass ``max_retries=0`` to run exactly once.

        Returns:
            Tuple[np.ndarray, list[ExtendedOptiStep]]: The optimized design and the
            history of the returned attempt. Each step is a :class:`ControlledOptiStep`
            and also carries the solver parameters and the phase it belongs to.
        """
        base_config = dataclasses.replace(self.config, **(config or {}))
        _validate_continuation(base_config.continuation)

        attempt_config = base_config
        reports: list[QualityReport] = []
        best: tuple[np.ndarray, list[ExtendedOptiStep], QualityReport] | None = None

        for attempt in range(max(0, base_config.max_retries) + 1):
            run_controller = controller if controller is not None else self.make_controller(attempt_config)
            design, history, report = self._run_attempt(attempt_config, run_controller, starting_point, attempt)
            reports.append(report)
            if best is None or _final_compliance(history) < _final_compliance(best[1]):
                best = (design, history, report)
            if report.passed:
                break
            attempt_config = _escalate(attempt_config, report)

        assert best is not None
        design, history, report = best
        self.quality_reports = tuple(reports)
        self.last_quality_report = report
        return design, history

    def _run_attempt(
        self,
        cfg: "Beams2D.Config",
        controller: ControllerLike,
        starting_point: npt.NDArray | None,
        attempt: int,
    ) -> tuple[np.ndarray, list[ExtendedOptiStep], QualityReport]:
        """Run one exploration phase plus sharpening tail, then evaluate the result."""
        nelx, nely = cfg.nelx, cfg.nely
        st = State.new(nelx, nely, cfg.rmin, cfg.forcedist)
        self.__st = st

        reset = getattr(controller, "reset", None)
        if callable(reset):
            reset()
        tail = _controller_tail(controller, cfg)

        run = _RunState(
            x=_initial_design(cfg, starting_point),
            signal=_controller_initial(controller, cfg),
            rmin=cfg.rmin,
        )
        run.x_print = _three_fields(st, run.x, cfg, run.signal.beta)[2]

        self._run_phase(cfg, st, controller, run, _exploration_budget(cfg), attempt, "exploration")

        if tail is not None and tail.iterations > 0:
            # Validity gate: the tail restarts from the best snapshot that reached the target
            # penalization while satisfying the volume constraint, or from uniform density.
            run.x = run.snapshot if run.snapshot is not None else cfg.volfrac * np.ones(nelx * nely)
            run.early_exit = False
            self._run_phase(cfg, st, FixedController(**_tail_kwargs(tail)), run, tail.iterations, attempt, "tail")

        report = evaluate(
            run.x_print.reshape(nelx, nely),
            st,
            volfrac=cfg.volfrac,
            compliance_history=run.compliance,
            early_exit=run.early_exit,
            attempt=attempt,
            connectivity_tol=cfg.connectivity_tol,
            grayness_tol=cfg.grayness_tol,
            volfrac_tol=cfg.volfrac_tol,
            compliance_ratio_tol=cfg.compliance_ratio_tol,
            stability_tol=cfg.stability_tol,
            stability_window=cfg.stability_window,
        )
        return design_to_image(run.x_print, nelx, nely), run.steps, report

    def _run_phase(  # noqa: PLR0913
        self,
        cfg: "Beams2D.Config",
        st: State,
        controller: ControllerLike,
        run: "_RunState",
        budget: int,
        attempt: int,
        phase: str,
    ) -> None:
        """Advance ``run`` by up to ``budget`` three-field SIMP iterations under ``controller``."""
        nelx, nely = cfg.nelx, cfg.nely
        gate = cfg.penal_gate if cfg.penal_gate is not None else cfg.penal
        loop = 0

        while loop < budget:
            run.signal = controller(_observe(cfg, run, budget, loop))
            if not np.isclose(run.signal.rmin, run.rmin):
                run.rmin = float(run.signal.rmin)
                st.H = h_mat(nelx, nely, run.rmin)
                st.Hs = st.H.sum(1)

            run.x_tilde, run.x_phys, run.x_print = _three_fields(st, run.x, cfg, run.signal.beta)
            step_cfg = dataclasses.replace(cfg, penal=run.signal.penal)
            run.ce = calc_sensitivity(run.x_print, st=st, cfg=dataclasses.asdict(step_cfg))
            self.reset_called = True  # override for multiple reset calls in optimize
            c = self.simulate(run.x_print, ce=run.ce, config=dataclasses.asdict(upcast(step_cfg, self.SimulateConfig)))

            compliance = float(c[0])
            run.steps.append(_make_step(np.array(c), len(run.steps), run.x_print, run.signal, attempt, phase))
            run.compliance.append(compliance)

            # Validity gate: remember the cheapest design that is already penalized and feasible.
            feasible = abs(float(np.mean(run.x_print)) - cfg.volfrac) <= cfg.volfrac_tol
            if phase != "tail" and run.signal.penal >= gate and feasible and compliance < run.snapshot_compliance:
                run.snapshot_compliance = compliance
                run.snapshot = run.x.copy()

            loop += 1

            dc, dv = _chain_sensitivities(st, cfg, run.signal, run.x_tilde, run.x_phys, run.x_print, run.ce)
            xnew = _oc_update(run.x, st, dc, dv, cfg, run.signal)
            run.change = float(np.max(np.abs(xnew - run.x)))
            run.x = xnew

            run.stagnation = run.stagnation + 1 if run.change <= st.min_change else 0
            if run.stagnation >= cfg.stagnation_patience:
                run.early_exit = True
                return
        run.early_exit = False

    # ----------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------

    def evaluate_quality(
        self,
        design: npt.NDArray,
        config: dict[str, Any] | None = None,
        *,
        compliance_history: list[float] | None = None,
        early_exit: bool = True,
    ) -> QualityReport:
        """Run the eight-check structural evaluator on an arbitrary design.

        Args:
            design (npt.NDArray): The design to check, as an image or a flat array.
            config (dict): A dictionary with configuration (e.g., boundary conditions).
            compliance_history (list[float] or None): Compliance per iteration, when
                available. Defaults to the compliance of ``design`` alone, which makes
                the compliance-ratio check trivially pass.
            early_exit (bool): Whether the run that produced ``design`` stopped on its
                change tolerance. Defaults to ``True`` for a standalone design.

        Returns:
            QualityReport: The five gating checks and the three informational metrics.
        """
        cfg = dataclasses.replace(self.config, **(config or {}))
        flat = image_to_design(design) if design.ndim > 1 else np.asarray(design, dtype=float)
        st = State.new(cfg.nelx, cfg.nely, cfg.rmin, cfg.forcedist)
        self.__st = st
        if not compliance_history:
            ce = calc_sensitivity(flat, st=st, cfg=dataclasses.asdict(cfg))
            compliance = float(((st.Emin + flat**cfg.penal * (st.Emax - st.Emin)) * ce).sum())
            compliance_history = [compliance]
        return evaluate(
            flat.reshape(cfg.nelx, cfg.nely),
            st,
            volfrac=cfg.volfrac,
            compliance_history=compliance_history,
            early_exit=early_exit,
            connectivity_tol=cfg.connectivity_tol,
            grayness_tol=cfg.grayness_tol,
            volfrac_tol=cfg.volfrac_tol,
            compliance_ratio_tol=cfg.compliance_ratio_tol,
            stability_tol=cfg.stability_tol,
            stability_window=cfg.stability_window,
        )

    def reset(self, seed: int | None = None, **kwargs) -> None:
        r"""Reset numpy random to a given seed and drop the cached quality reports.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        super().reset(seed, **kwargs)
        self.__st = State()
        self.last_quality_report = None
        self.quality_reports = ()


@dataclass
class _RunState:
    """Mutable state carried across the phases of one solver attempt.

    Attributes:
        x: The current design field.
        signal: The control signal of the last iteration.
        rmin: The filter radius the assembled filter matrix currently uses.
        x_tilde: The filtered field of the last iteration.
        x_phys: The projected physical field of the last iteration.
        x_print: The printed field of the last *evaluated* design.
        ce: Element-wise strain energy of the last evaluated design.
        steps: The optimization history, across all phases.
        compliance: The compliance recorded at every iteration.
        change: Infinity norm of the last design-variable update.
        stagnation: Consecutive iterations with ``change`` below the tolerance.
        early_exit: Whether the last phase stopped before exhausting its budget.
        snapshot: The best design field that passed the validity gate.
        snapshot_compliance: The compliance of ``snapshot``.
    """

    x: npt.NDArray[np.float64]
    signal: ControlSignal
    rmin: float
    x_tilde: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    x_phys: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    x_print: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    ce: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    steps: list[ExtendedOptiStep] = field(default_factory=list)
    compliance: list[float] = field(default_factory=list)
    change: float = 1.0
    stagnation: int = 0
    early_exit: bool = False
    snapshot: npt.NDArray[np.float64] | None = None
    snapshot_compliance: float = float("inf")


def _exploration_budget(cfg: "Beams2D.Config") -> int:
    """Return the iterations left for exploration once the tail is carved out of ``max_iter``."""
    return max(0, cfg.max_iter - _tail_iterations(cfg))


def _tail_iterations(cfg: "Beams2D.Config") -> int:
    """Return the length of the sharpening tail, never more than half the budget."""
    if cfg.continuation == "fixed":
        return 0
    return max(0, min(cfg.tail_iters, cfg.max_iter // 2))


def _make_tail(cfg: "Beams2D.Config") -> TailSpec | None:
    """Build the sharpening tail for ``cfg``, or ``None`` when the budget leaves no room."""
    iterations = _tail_iterations(cfg)
    if iterations == 0:
        return None
    return TailSpec(
        penal=cfg.tail_penal if cfg.tail_penal is not None else cfg.penal,
        rmin=cfg.tail_rmin if cfg.tail_rmin is not None else cfg.rmin,
        iterations=iterations,
        beta=cfg.tail_beta,
        move=cfg.tail_move,
    )


def _tail_kwargs(tail: TailSpec) -> dict[str, float]:
    """Return the :class:`FixedController` arguments that hold ``tail`` constant."""
    return {"penal": tail.penal, "beta": tail.beta, "rmin": tail.rmin, "move": tail.move}


def _controller_initial(controller: ControllerLike, cfg: "Beams2D.Config") -> ControlSignal:
    """Ask ``controller`` for its starting parameters, falling back to the configured ones."""
    initialize = getattr(controller, "initialize", None)
    if callable(initialize):
        signal = initialize()
        if signal is not None:
            return signal
    return ControlSignal(penal=cfg.penal_init, beta=cfg.beta_init, rmin=cfg.rmin, move=cfg.move_init)


def _controller_tail(controller: ControllerLike, cfg: "Beams2D.Config") -> TailSpec | None:
    """Ask ``controller`` for its sharpening tail; a plain callable gets the configured one."""
    finalize = getattr(controller, "finalize", None)
    if callable(finalize):
        return finalize()
    return _make_tail(cfg)


def _final_compliance(history: list[ExtendedOptiStep]) -> float:
    """Return the last recorded compliance of ``history``, or infinity when it is empty."""
    return float(history[-1].obj_values[0]) if history else float("inf")


def _initial_design(cfg: "Beams2D.Config", starting_point: npt.NDArray | None) -> npt.NDArray[np.float64]:
    """Build the initial design field, either uniform or warm-started from a given design."""
    if starting_point is None:
        return cfg.volfrac * np.ones(cfg.nelx * cfg.nely, dtype=float)
    flat = np.asarray(image_to_design(starting_point), dtype=float)
    eps = 1e-4
    # Add tiny non-zero values to avoid warm-start gradient issues for zero values.
    return np.clip((1 - eps) * flat + eps * cfg.volfrac, 0.0, 1.0)


def _three_fields(
    st: State, x: npt.NDArray[np.float64], cfg: "Beams2D.Config", beta: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Map the design field through the three-field chain.

    Args:
        st: State holding the density filter.
        x: The design field :math:`x`.
        cfg: The configuration of the run (supplies ``eta`` and the overhang flag).
        beta: The projection sharpness :math:`\beta` of the current iteration.

    Returns:
        Tuple of the filtered field :math:`\tilde{x}`, the projected physical field
        :math:`\bar{x}` and the printed field that the physics sees.
    """
    x_tilde = density_filter(st, x)
    x_phys = heaviside_projection(x_tilde, beta, cfg.eta)
    x_print = overhang_filter_x(x_phys.reshape(cfg.nelx, cfg.nely)) if cfg.overhang_constraint else x_phys
    return x_tilde, x_phys, x_print


def _observe(cfg: "Beams2D.Config", run: "_RunState", budget: int, loop: int) -> Observation:
    """Assemble the observation the continuation controller sees before iteration ``loop``."""
    return Observation(
        iteration=loop,
        max_iter=budget,
        compliance=run.compliance[-1] if run.compliance else float("inf"),
        best_compliance=min(run.compliance) if run.compliance else float("inf"),
        grayness=grayness(run.x_print),
        volume_fraction=float(np.mean(run.x_print)),
        checkerboard=checkerboard_index(run.x_print.reshape(cfg.nelx, cfg.nely)),
        change=run.change,
        stagnation=run.stagnation,
        budget_used=loop / max(1, budget),
        signal=run.signal,
    )


def _make_step(  # noqa: PLR0913
    obj_values: npt.NDArray[np.float64],
    loop: int,
    x_print: npt.NDArray[np.float64],
    signal: ControlSignal,
    attempt: int,
    phase: str,
) -> ControlledOptiStep:
    """Record one optimization step together with the control signal that produced it."""
    step = ControlledOptiStep(obj_values=obj_values, step=loop)
    step.design = np.array(x_print)
    step.penal = signal.penal
    step.beta = signal.beta
    step.rmin = signal.rmin
    step.move = signal.move
    step.attempt = attempt
    step.phase = phase
    return step


def _chain_sensitivities(  # noqa: PLR0913
    st: State,
    cfg: "Beams2D.Config",
    signal: ControlSignal,
    x_tilde: npt.NDArray[np.float64],
    x_phys: npt.NDArray[np.float64],
    x_print: npt.NDArray[np.float64],
    ce: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Chain the compliance and volume sensitivities back to the design field.

    The sensitivities start out w.r.t. the printed field and are pulled through the
    overhang filter (when enabled), the Heaviside projection and the density filter,
    which is what makes the three-field optimizer consistent with its own physics.

    Args:
        st: State holding the filter and the stiffness bounds.
        cfg: The configuration of the run.
        signal: The control signal of the current iteration.
        x_tilde: The filtered field.
        x_phys: The projected physical field.
        x_print: The printed field.
        ce: Element-wise strain energy of the printed field.

    Returns:
        Tuple of the non-positive compliance sensitivity and the positive volume
        sensitivity, both w.r.t. the design field.
    """
    dc = (-signal.penal * x_print ** (signal.penal - 1) * (st.Emax - st.Emin)) * ce
    dv = np.ones(cfg.nelx * cfg.nely)
    if cfg.overhang_constraint:
        _, dc, dv = overhang_filter_d(x_phys.reshape(cfg.nelx, cfg.nely), dc, dv)
    dproj = heaviside_derivative(x_tilde, signal.beta, cfg.eta)
    dc = np.clip(filter_sensitivity(st, dc * dproj), None, 0.0)
    dv = np.maximum(filter_sensitivity(st, dv * dproj), 1e-12)
    return dc, dv


def _oc_update(  # noqa: PLR0913
    x: npt.NDArray[np.float64],
    st: State,
    dc: npt.NDArray[np.float64],
    dv: npt.NDArray[np.float64],
    cfg: "Beams2D.Config",
    signal: ControlSignal,
) -> npt.NDArray[np.float64]:
    """Optimality-criteria update of the design field under the three-field volume constraint.

    The bisection on the Lagrange multiplier compares the volume of the *printed*
    field -- filtered, projected and, if requested, overhang-filtered -- against
    the target, so the constraint is enforced on the density that the physics and
    the returned design actually use.

    Args:
        x: The current design field.
        st: State holding the filter and the stopping tolerances.
        dc: Compliance sensitivity w.r.t. the design field (non-positive).
        dv: Volume sensitivity w.r.t. the design field (positive).
        cfg: The configuration of the run.
        signal: The control signal of the current iteration (supplies the move limit and sharpness).

    Returns:
        npt.NDArray: The updated design field.
    """
    n = cfg.nelx * cfg.nely
    move = signal.move
    l1, l2 = 0.0, 1e9
    xnew = x.copy()

    while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > st.min_ratio:
        lmid = 0.5 * (l2 + l1)
        if lmid > 0:
            xnew = np.maximum(
                0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
            )
        else:
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

        _, _, xPrint = _three_fields(st, xnew, cfg, signal.beta)

        if xPrint.sum() > cfg.volfrac * n:
            l1 = lmid
        else:
            l2 = lmid

        # Ensures this loop does not become stuck due to abs(l2 - l1) converging to near 0
        if abs(l2 - l1) < np.finfo(float).eps:
            break

    return xnew


#: Upper bound on the iteration budget granted by the closed-loop retries.
_MAX_ITER_CAP = 1000


def _escalate(cfg: "Beams2D.Config", report: QualityReport) -> "Beams2D.Config":
    """Derive the settings of the next attempt from the checks that just failed.

    Implements the rerun hints of the paper: a volume violation adjusts the
    volume-fraction target by the observed deviation, and every other failure --
    convergence, grayness, or no specific hint at all -- grows the iteration
    budget by 30%.

    Args:
        cfg: The configuration of the attempt that failed.
        report: The report of that attempt.

    Returns:
        Beams2D.Config: The escalated configuration.
    """
    failed = set(report.failed_checks)
    updates: dict[str, Any] = {}

    if "volume_fraction" in failed:
        deviation = report.metrics["volume_fraction"] - cfg.volfrac
        updates["volfrac"] = float(np.clip(cfg.volfrac - deviation, 0.01, 0.99))
    if failed - {"volume_fraction"}:
        updates["max_iter"] = min(round(cfg.max_iter * 1.3) + 1, _MAX_ITER_CAP)

    return dataclasses.replace(cfg, **updates)


if __name__ == "__main__":
    main(Beams2D)
