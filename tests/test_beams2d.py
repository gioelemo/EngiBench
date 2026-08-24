"""Tests for the AutoSiMP three-field solver of the Beams2D problem."""

import dataclasses
from itertools import pairwise
import math

import numpy as np
import pytest

pytest.importorskip("cvxopt")
pytest.importorskip("scipy")

from engibench.problems.beams2d.autosimp import checkerboard_index
from engibench.problems.beams2d.autosimp import connectivity_fraction
from engibench.problems.beams2d.autosimp import ControlSignal
from engibench.problems.beams2d.autosimp import density_filter
from engibench.problems.beams2d.autosimp import evaluate
from engibench.problems.beams2d.autosimp import filter_sensitivity
from engibench.problems.beams2d.autosimp import FixedController
from engibench.problems.beams2d.autosimp import grayness
from engibench.problems.beams2d.autosimp import heaviside_derivative
from engibench.problems.beams2d.autosimp import heaviside_projection
from engibench.problems.beams2d.autosimp import load_elements
from engibench.problems.beams2d.autosimp import load_path_efficiency
from engibench.problems.beams2d.autosimp import Observation
from engibench.problems.beams2d.autosimp import reachable_from_supports
from engibench.problems.beams2d.autosimp import ScheduleController
from engibench.problems.beams2d.autosimp import support_elements
from engibench.problems.beams2d.autosimp import TailSpec
from engibench.problems.beams2d.autosimp import thin_member_fraction
from engibench.problems.beams2d.autosimp import ThreeFieldController
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import State
from engibench.problems.beams2d.v2 import Beams2D
from engibench.problems.beams2d.v2 import ControlledOptiStep

# A budget on which the default controller passes every gate on its first attempt.
SMALL = {"nelx": 30, "nely": 15, "max_iter": 80, "rmin": 1.5}
N_CHECKS = 8
N_GATING_CHECKS = 5


def _controlled(history: list) -> list[ControlledOptiStep]:
    """Narrow an optimization history to the steps that carry a control signal."""
    steps = [step for step in history if isinstance(step, ControlledOptiStep)]
    assert len(steps) == len(history), "every v2 step must record its control signal"
    return steps


# ------------------------------------------------------------------------------------------
# Heaviside projection
# ------------------------------------------------------------------------------------------


def test_projection_maps_unit_interval_onto_itself() -> None:
    x = np.linspace(0.0, 1.0, 101)
    for beta in (1.0, 4.0, 16.0, 64.0):
        projected = heaviside_projection(x, beta)
        assert projected[0] == pytest.approx(0.0)
        assert projected[-1] == pytest.approx(1.0)
        assert np.all(np.diff(projected) >= 0.0), "the projection must stay monotone"
        assert np.all((projected >= 0.0) & (projected <= 1.0))
        # Strictly increasing wherever the tanh has not saturated in double precision.
        near_threshold = projected[45:56]
        assert np.all(np.diff(near_threshold) > 0.0)


def test_projection_fixes_the_threshold_and_sharpens_with_beta() -> None:
    x = np.array([0.2, 0.5, 0.8])
    mild = heaviside_projection(x, 1.0)
    sharp = heaviside_projection(x, 32.0)
    assert mild[1] == pytest.approx(0.5)
    assert sharp[1] == pytest.approx(0.5)
    assert sharp[0] < mild[0], "a sharper projection pushes sub-threshold densities towards void"
    assert sharp[2] > mild[2], "a sharper projection pushes super-threshold densities towards solid"


def test_projection_is_the_identity_for_vanishing_beta() -> None:
    x = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(heaviside_projection(x, 0.0), x)
    np.testing.assert_allclose(heaviside_derivative(x, 0.0), np.ones_like(x))


def test_projection_derivative_matches_finite_differences() -> None:
    x = np.linspace(0.05, 0.95, 19)
    h = 1e-6
    for beta in (1.0, 8.0, 32.0):
        fd = (heaviside_projection(x + h, beta) - heaviside_projection(x - h, beta)) / (2 * h)
        np.testing.assert_allclose(heaviside_derivative(x, beta), fd, rtol=1e-5, atol=1e-8)


# ------------------------------------------------------------------------------------------
# Filter and the three-field sensitivity chain
# ------------------------------------------------------------------------------------------


def test_filter_sensitivity_is_the_adjoint_of_the_density_filter() -> None:
    st = State.new(nelx=8, nely=5, rmin=2.0, forcedist=0.0)
    rng = np.random.default_rng(0)
    u = rng.normal(size=8 * 5)
    v = rng.normal(size=8 * 5)
    # <filter(u), v> == <u, filter_sensitivity(v)>
    assert np.dot(density_filter(st, u), v) == pytest.approx(np.dot(u, filter_sensitivity(st, v)))


def test_three_field_sensitivities_match_finite_differences() -> None:
    nelx, nely, penal, beta, eta = 12, 6, 3.0, 4.0, 0.5
    st = State.new(nelx=nelx, nely=nely, rmin=1.8, forcedist=0.0)
    cfg = {"nelx": nelx, "nely": nely, "penal": penal}
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.8, nelx * nely)

    def compliance(design: np.ndarray) -> float:
        x_phys = heaviside_projection(density_filter(st, design), beta, eta)
        ce = calc_sensitivity(x_phys, st=st, cfg=cfg)
        return float(((st.Emin + x_phys**penal * (st.Emax - st.Emin)) * ce).sum())

    def volume(design: np.ndarray) -> float:
        return float(heaviside_projection(density_filter(st, design), beta, eta).sum())

    x_tilde = density_filter(st, x)
    x_phys = heaviside_projection(x_tilde, beta, eta)
    ce = calc_sensitivity(x_phys, st=st, cfg=cfg)
    dproj = heaviside_derivative(x_tilde, beta, eta)
    dc = filter_sensitivity(st, ((-penal * x_phys ** (penal - 1) * (st.Emax - st.Emin)) * ce) * dproj)
    dv = filter_sensitivity(st, np.ones(nelx * nely) * dproj)

    h = 1e-6
    for i in rng.choice(nelx * nely, 6, replace=False):
        plus, minus = x.copy(), x.copy()
        plus[i] += h
        minus[i] -= h
        assert dc[i] == pytest.approx((compliance(plus) - compliance(minus)) / (2 * h), rel=1e-4)
        assert dv[i] == pytest.approx((volume(plus) - volume(minus)) / (2 * h), rel=1e-6)


# ------------------------------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------------------------------


def test_grayness_is_zero_for_binary_and_one_for_uniform_gray() -> None:
    assert grayness(np.array([0.0, 1.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert grayness(np.full(10, 0.5)) == pytest.approx(1.0)
    assert grayness(np.array([])) == pytest.approx(0.0)


def test_checkerboard_index_separates_a_checkerboard_from_a_solid_block() -> None:
    checkerboard = np.indices((8, 8)).sum(axis=0) % 2
    assert checkerboard_index(checkerboard.astype(float)) == pytest.approx(1.0)
    assert checkerboard_index(np.ones((8, 8))) == pytest.approx(0.0)
    # A linear ramp has no alternating component either.
    assert checkerboard_index(np.tile(np.linspace(0, 1, 8), (8, 1))) == pytest.approx(0.0)
    assert checkerboard_index(np.ones((1, 8))) == pytest.approx(0.0)


def test_thin_member_fraction_counts_one_element_wide_connections() -> None:
    thick = np.zeros((20, 20))
    thick[:, 5:15] = 1.0
    assert thin_member_fraction(thick) == pytest.approx(0.0)

    # A single-column bar is one-element-wide everywhere.
    thin = np.zeros((20, 20))
    thin[:, 10] = 1.0
    assert thin_member_fraction(thin) == pytest.approx(1.0)

    # A two-column bar is not, not even at the domain boundary.
    two_wide = np.zeros((20, 20))
    two_wide[:, :2] = 1.0
    assert thin_member_fraction(two_wide) == pytest.approx(0.0)

    # Half thin, half thick.
    mixed = np.zeros((20, 20))
    mixed[:10, 5:15] = 1.0  # 100 solid elements, none thin
    mixed[15, :10] = 1.0  # 10 solid elements, all thin
    assert thin_member_fraction(mixed) == pytest.approx(10 / 110)

    assert thin_member_fraction(np.zeros((5, 5))) == pytest.approx(0.0)


def test_connectivity_fraction_measures_material_reachable_from_the_supports() -> None:
    nelx, nely = 20, 10
    st = State.new(nelx=nelx, nely=nely, rmin=1.5, forcedist=0.0)
    supports = support_elements(st, nelx, nely)
    loads = load_elements(st, nelx, nely)
    assert supports.any()
    assert loads.any()

    solid = np.ones((nelx, nely))
    assert connectivity_fraction(solid, supports) == pytest.approx(1.0)
    assert reachable_from_supports(solid, supports).all()

    # An island of material detached from the supports lowers the fraction.
    island = np.zeros((nelx, nely))
    island[0, :] = 1.0  # touches the left-edge symmetry support: 10 elements
    island[10:12, 4:6] = 1.0  # floating 2x2 block: 4 elements
    assert connectivity_fraction(island, supports) == pytest.approx(10 / 14)

    assert connectivity_fraction(np.zeros((nelx, nely)), supports) == pytest.approx(0.0)


def test_load_path_efficiency_is_the_detour_factor_of_the_load_path() -> None:
    nelx, nely = 20, 10
    st = State.new(nelx=nelx, nely=nely, rmin=1.5, forcedist=0.0)
    supports = support_elements(st, nelx, nely)
    loads = load_elements(st, nelx, nely)

    # The load element (0, 0) already touches the left-edge support: zero-length path.
    solid = np.ones((nelx, nely))
    assert load_path_efficiency(solid, supports, loads) == pytest.approx(1.0)

    # No solid material at all: nothing to carry the load.
    assert load_path_efficiency(np.zeros((nelx, nely)), supports, loads) == math.inf

    # A straight 4-connected path is its own Euclidean distance, so the ratio is 1.
    strip = np.zeros((nelx, nely))
    strip[:, 0] = 1.0
    assert load_path_efficiency(strip, supports, loads) == pytest.approx(1.0)


def test_support_and_load_elements_sit_on_the_expected_boundaries() -> None:
    nelx, nely = 20, 10
    st = State.new(nelx=nelx, nely=nely, rmin=1.5, forcedist=0.0)
    supports = support_elements(st, nelx, nely)
    loads = load_elements(st, nelx, nely)
    # Symmetry condition on the whole left edge plus the bottom-right roller.
    assert supports[0, :].all()
    assert supports[-1, -1]
    assert not supports[nelx // 2, nely // 2]
    # The default load sits on the top-left node.
    assert loads[0, 0]
    assert loads.sum() == 1


# ------------------------------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------------------------------


def _report(x_2d: np.ndarray, st: State, **kwargs):
    defaults = {
        "volfrac": float(np.mean(x_2d)),
        "compliance_history": [10.0] * 12,
        "early_exit": True,
    }
    return evaluate(x_2d, st, **{**defaults, **kwargs})


def test_evaluator_reports_five_gates_and_three_informational_metrics() -> None:
    nelx, nely = 20, 10
    st = State.new(nelx=nelx, nely=nely, rmin=1.5, forcedist=0.0)
    report = _report(np.ones((nelx, nely)), st)
    assert len(report.checks) == N_CHECKS
    gating = [c for c in report.checks.values() if not c.informational]
    informational = [c for c in report.checks.values() if c.informational]
    assert len(gating) == N_GATING_CHECKS
    assert len(informational) == N_CHECKS - N_GATING_CHECKS
    assert report.passed
    assert report.failed_checks == []
    assert set(report.metrics) == set(report.checks)
    assert report.to_dict()["passed"] is True
    assert "connectivity" in str(report)


def test_evaluator_fails_disconnected_gray_and_infeasible_designs() -> None:
    nelx, nely = 20, 10
    st = State.new(nelx=nelx, nely=nely, rmin=1.5, forcedist=0.0)

    disconnected = np.zeros((nelx, nely))
    assert "connectivity" in _report(disconnected, st, volfrac=0.5).failed_checks

    # A block of material that touches neither the left-edge symmetry support nor the roller.
    floating = np.zeros((nelx, nely))
    floating[0, :] = 1.0
    floating[8:14, 3:7] = 1.0
    assert "connectivity" in _report(floating, st, volfrac=float(np.mean(floating))).failed_checks

    gray = np.full((nelx, nely), 0.5)
    failed = _report(gray, st, volfrac=0.5).failed_checks
    assert "grayness" in failed

    solid = np.ones((nelx, nely))
    # Absolute deviation, as in the paper: |1.0 - 0.5| = 0.5 > 0.02.
    assert "volume_fraction" in _report(solid, st, volfrac=0.5).failed_checks
    # A 1% absolute deviation passes the default 2% gate even though it is 2% relative.
    near = np.zeros((nelx, nely))
    near.reshape(-1)[: int(0.51 * nelx * nely)] = 1.0
    assert "volume_fraction" not in _report(near, st, volfrac=0.5).failed_checks
    # Tail degradation: the final compliance is far above the best one seen.
    tail = _report(solid, st, compliance_history=[10.0, 10.0, 100.0])
    assert "compliance_ratio" in tail.failed_checks
    # Not converged: no early exit and a wildly oscillating tail.
    noisy = _report(solid, st, early_exit=False, compliance_history=[10.0, 30.0, 10.0, 30.0] * 3)
    assert "convergence" in noisy.failed_checks


# ------------------------------------------------------------------------------------------
# Continuation controllers
# ------------------------------------------------------------------------------------------


def _observation(iteration: int, max_iter: int, stagnation: int = 0) -> Observation:
    return Observation(
        iteration=iteration,
        max_iter=max_iter,
        compliance=1.0,
        best_compliance=1.0,
        grayness=0.5,
        volume_fraction=0.35,
        checkerboard=0.0,
        change=1.0,
        stagnation=stagnation,
        budget_used=iteration / max_iter,
        signal=ControlSignal(penal=1.0, beta=1.0, rmin=2.0, move=0.2),
    )


def test_fixed_controller_holds_every_parameter_constant() -> None:
    controller = FixedController(penal=3.0, beta=2.0, rmin=2.0, move=0.2)
    signals = [controller(_observation(i, 50)) for i in range(50)]
    assert all(s == signals[0] for s in signals)


def test_schedule_controller_walks_its_four_phases() -> None:
    max_iter = 100
    controller = ScheduleController(penal_target=3.0, rmin=2.0, max_iter=max_iter, beta_max=16.0)
    signals = [controller(_observation(i, max_iter)) for i in range(max_iter)]

    penals = [s.penal for s in signals]
    betas = [s.beta for s in signals]
    moves = [s.move for s in signals]

    assert penals[0] == pytest.approx(1.0)
    assert penals[-1] == pytest.approx(3.0)
    assert betas[0] == pytest.approx(1.0)
    assert betas[-1] == pytest.approx(16.0)
    assert all(b in {1.0, 2.0, 4.0, 8.0, 16.0} for b in betas), "beta continuation proceeds by doubling"
    assert all(a <= b for a, b in pairwise(penals)), "the penalization never decreases"
    assert all(a <= b for a, b in pairwise(betas)), "the sharpness never decreases"

    # Exploration: neither continuation has started yet.
    explore = int(controller.exploration_end * max_iter) - 1
    assert penals[explore] == pytest.approx(1.0)
    assert betas[explore] == pytest.approx(1.0)
    # Penalization: penal reaches its target by the end of the phase, beta still held down.
    penalize = int(controller.penalization_end * max_iter)
    assert penals[penalize] == pytest.approx(3.0)
    assert betas[penalize] == pytest.approx(1.0)
    # Sharpening then convergence: beta tops out and the move limit drops.
    assert betas[int(controller.sharpening_end * max_iter)] == pytest.approx(16.0)
    assert moves[0] == pytest.approx(0.2)
    assert moves[-1] == pytest.approx(0.05)


def test_schedule_controller_reports_its_start_and_its_tail() -> None:
    tail = TailSpec(penal=3.0, rmin=2.0, iterations=40)
    controller = ScheduleController(penal_target=3.0, rmin=2.0, max_iter=100, tail=tail)
    start = controller.initialize()
    assert start.penal == pytest.approx(1.0)
    assert start.beta == pytest.approx(1.0)
    assert controller.finalize() is tail
    assert tail.signal() == ControlSignal(penal=3.0, beta=32.0, rmin=2.0, move=0.05)
    assert ScheduleController(penal_target=3.0, rmin=2.0, max_iter=100).finalize() is None


def test_fixed_controller_runs_no_tail() -> None:
    assert FixedController(penal=3.0, beta=1.0, rmin=2.0, move=0.2).finalize() is None


def test_three_field_controller_matches_the_documented_baseline() -> None:
    max_iter = 100
    controller = ThreeFieldController(
        penal_target=4.5, rmin=2.0, max_iter=max_iter, penal_ramp=30, beta_interval=10, beta_max=16.0
    )
    signals = [controller(_observation(i, max_iter)) for i in range(max_iter)]
    # Linear penal ramp from 1.0 to 4.5 over 30 iterations, then held.
    assert signals[0].penal == pytest.approx(1.0)
    assert signals[15].penal == pytest.approx(1.0 + 0.5 * 3.5)
    assert signals[30].penal == pytest.approx(4.5)
    assert signals[-1].penal == pytest.approx(4.5)
    # Geometric beta doubling every 10 iterations, capped at beta_max.
    assert [signals[i].beta for i in (0, 10, 20, 30, 40)] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert signals[-1].beta == pytest.approx(16.0)
    # No late tightening unless a final radius is asked for.
    assert all(s.rmin == pytest.approx(2.0) for s in signals)
    tightening = ThreeFieldController(penal_target=4.5, rmin=2.0, max_iter=max_iter, rmin_final=1.2)
    assert tightening(_observation(0, max_iter)).rmin == pytest.approx(2.0)
    assert tightening(_observation(max_iter - 1, max_iter)).rmin == pytest.approx(1.2)


def test_schedule_controller_sharpens_early_on_stagnation() -> None:
    max_iter = 100
    controller = ScheduleController(penal_target=3.0, rmin=2.0, max_iter=max_iter, stagnation_trigger=2)
    baseline = ScheduleController(penal_target=3.0, rmin=2.0, max_iter=max_iter, stagnation_trigger=2)
    for i in range(10):
        controller(_observation(i, max_iter, stagnation=5))
        baseline(_observation(i, max_iter, stagnation=0))
    assert controller(_observation(10, max_iter)).beta > baseline(_observation(10, max_iter)).beta

    controller.reset()
    assert controller(_observation(10, max_iter)).beta == baseline(_observation(10, max_iter)).beta


def test_schedule_controller_handles_a_degenerate_budget() -> None:
    controller = ScheduleController(penal_target=3.0, rmin=2.0, max_iter=1, beta_init=8.0, beta_max=8.0)
    assert controller.n_doublings == 0
    signal = controller(_observation(0, 1))
    assert signal.beta == pytest.approx(8.0)
    assert signal.penal == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------
# The problem itself
# ------------------------------------------------------------------------------------------


def test_v2_config_rejects_an_unknown_continuation_mode() -> None:
    with pytest.raises(AssertionError):
        Beams2D(config={**SMALL, "continuation": "magic"})


def test_v2_keeps_the_condition_and_simulate_configs_intact() -> None:
    problem = Beams2D(config=SMALL)
    assert type(problem.conditions) is Beams2D.Conditions
    assert type(problem.simulate_config) is Beams2D.SimulateConfig
    # The new knobs live on `Config` only, so the dataset schema is untouched.
    condition_fields = {f.name for f in dataclasses.fields(Beams2D.Conditions)}
    assert "beta_max" not in condition_fields
    assert {f.name for f in dataclasses.fields(Beams2D.Conditions)} == {
        "volfrac",
        "rmin",
        "forcedist",
        "overhang_constraint",
    }


def test_v2_optimize_produces_a_feasible_near_binary_design() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    design, history = problem.optimize()

    assert design.shape == (SMALL["nely"], SMALL["nelx"])
    assert np.all((design >= 0.0) & (design <= 1.0))
    assert design.mean() == pytest.approx(Beams2D.Conditions.volfrac, abs=0.01)
    assert history, "the optimizer must report its history"
    assert [step.step for step in _controlled(history)] == list(range(len(history)))

    report = problem.last_quality_report
    assert report is not None
    assert report.passed, f"unexpected failures: {report.failed_checks}"
    assert grayness(design) <= problem.config.grayness_tol


def test_v2_records_the_control_signal_of_every_step() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    _, history = problem.optimize()
    steps = _controlled(history)
    penals = [step.penal for step in steps]
    betas = [step.beta for step in steps]
    assert penals[0] < penals[-1], "the penalization continuation must have advanced"
    assert betas[0] < betas[-1], "the sharpness continuation must have advanced"
    assert all(step.rmin == pytest.approx(SMALL["rmin"]) for step in steps)


def test_v2_beats_v1_on_discreteness() -> None:
    v1 = pytest.importorskip("engibench.problems.beams2d.v1")
    design_v1, _ = v1.Beams2D(seed=0, config=SMALL).optimize()
    design_v2, _ = Beams2D(seed=0, config=SMALL).optimize()
    assert grayness(design_v2) < grayness(design_v1), (
        "the Heaviside projection must yield a more discrete design than the plain density filter"
    )


def test_v2_accepts_a_custom_direct_numeric_controller() -> None:
    seen: list[Observation] = []

    def controller(observation: Observation) -> ControlSignal:
        seen.append(observation)
        return ControlSignal(penal=2.0, beta=3.0, rmin=SMALL["rmin"], move=0.15)

    # No tail, so the custom controller drives every single iteration.
    problem = Beams2D(seed=0, config={**SMALL, "max_iter": 10, "max_retries": 0, "tail_iters": 0})
    _, history = problem.optimize(controller=controller)

    steps = _controlled(history)
    assert len(seen) == len(steps)
    assert seen[0].iteration == 0
    assert seen[0].max_iter == 10  # noqa: PLR2004
    assert all(step.penal == pytest.approx(2.0) for step in steps)
    assert all(step.beta == pytest.approx(3.0) for step in steps)
    assert all(step.move == pytest.approx(0.15) for step in steps)
    assert {step.phase for step in steps} == {"exploration"}


def test_v2_gives_a_plain_callable_the_shared_sharpening_tail() -> None:
    def controller(observation: Observation) -> ControlSignal:
        del observation
        return ControlSignal(penal=2.0, beta=3.0, rmin=SMALL["rmin"], move=0.15)

    problem = Beams2D(seed=0, config={**SMALL, "max_iter": 20, "max_retries": 0, "tail_iters": 6})
    _, history = problem.optimize(controller=controller)
    steps = _controlled(history)

    exploration = [s for s in steps if s.phase == "exploration"]
    tail = [s for s in steps if s.phase == "tail"]
    assert len(exploration) == 14  # noqa: PLR2004
    assert len(tail) == 6  # noqa: PLR2004
    assert all(s.beta == pytest.approx(3.0) for s in exploration)
    assert all(s.beta == pytest.approx(problem.config.tail_beta) for s in tail)
    assert all(s.move == pytest.approx(problem.config.tail_move) for s in tail)


def test_v2_tail_is_clamped_to_half_the_budget() -> None:
    problem = Beams2D(seed=0, config={**SMALL, "max_iter": 20, "max_retries": 0, "tail_iters": 40})
    _, history = problem.optimize()
    steps = _controlled(history)
    assert sum(s.phase == "tail" for s in steps) == 10  # noqa: PLR2004
    assert sum(s.phase == "exploration" for s in steps) == 10  # noqa: PLR2004


def test_v2_retries_when_a_gate_fails() -> None:
    # An impossible grayness target forces the closed loop to exhaust its retries.
    problem = Beams2D(seed=0, config={**SMALL, "max_iter": 8, "grayness_tol": 0.0, "max_retries": 2})
    problem.optimize()
    assert len(problem.quality_reports) == 3  # noqa: PLR2004
    assert [r.attempt for r in problem.quality_reports] == [0, 1, 2]
    report = problem.last_quality_report
    assert report is not None
    assert not report.passed
    assert "grayness" in report.failed_checks


def test_v2_does_not_retry_when_the_first_attempt_passes() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    problem.optimize()
    assert problem.quality_reports[0].passed
    assert len(problem.quality_reports) == 1


def test_v2_rerun_hint_grows_the_budget_by_thirty_percent() -> None:
    from engibench.problems.beams2d.v2 import _escalate  # noqa: PLC0415

    problem = Beams2D(config={**SMALL, "max_iter": 100})
    gray = problem.evaluate_quality(np.full((SMALL["nely"], SMALL["nelx"]), 0.35), config={"volfrac": 0.35})
    assert "grayness" in gray.failed_checks
    escalated = _escalate(problem.config, gray)
    assert escalated.max_iter == round(100 * 1.3) + 1
    assert escalated.volfrac == pytest.approx(problem.config.volfrac), "no volume hint, no volume change"


def test_v2_rerun_hint_adjusts_the_volume_target() -> None:
    from engibench.problems.beams2d.v2 import _escalate  # noqa: PLC0415

    cfg = Beams2D(config=SMALL).config
    report = Beams2D(config=SMALL).evaluate_quality(np.ones((SMALL["nely"], SMALL["nelx"])))
    assert "volume_fraction" in report.failed_checks
    escalated = _escalate(cfg, report)
    # The design came out at 1.0 against a 0.35 target, so the target is pulled down.
    assert escalated.volfrac < cfg.volfrac
    assert escalated.volfrac == pytest.approx(np.clip(cfg.volfrac - (1.0 - cfg.volfrac), 0.01, 0.99))


def test_v2_warm_start_keeps_the_design_feasible() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    design, _ = problem.optimize()
    warm, history = problem.optimize(starting_point=design, config={"max_iter": 10, "max_retries": 0})
    assert warm.shape == design.shape
    assert warm.mean() == pytest.approx(0.35, abs=0.01)
    assert len(history) <= 10  # noqa: PLR2004


def test_v2_supports_the_overhang_constraint() -> None:
    problem = Beams2D(seed=0, config={**SMALL, "overhang_constraint": True, "max_iter": 20, "max_retries": 0})
    design, _ = problem.optimize()
    assert design.mean() == pytest.approx(0.35, abs=0.01)
    assert np.all(np.isfinite(design))


def test_v2_evaluate_quality_works_on_a_standalone_design() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    design, _ = problem.optimize()
    report = problem.evaluate_quality(design)
    assert report.passed, f"unexpected failures: {report.failed_checks}"
    assert 0.0 <= report.metrics["load_path_efficiency"] <= 1.0

    void = np.zeros((SMALL["nely"], SMALL["nelx"]))
    assert "connectivity" in problem.evaluate_quality(void).failed_checks


def test_v2_reset_clears_the_quality_reports() -> None:
    problem = Beams2D(seed=0, config=SMALL)
    problem.optimize()
    assert problem.last_quality_report is not None
    problem.reset(seed=1)
    assert problem.last_quality_report is None
    assert problem.quality_reports == ()
