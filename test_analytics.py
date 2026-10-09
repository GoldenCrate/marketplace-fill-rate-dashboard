"""
test_analytics.py — pytest suite for analytics.py.

Covers KPI math, funnel construction & monotonicity, fill-rate slicing,
unfilled-segment flagging, and the two-proportion z-test (including a
known-value check against hand-computed numbers and the A/B wrapper on the
synthetic data).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import analytics as A
from data_gen import generate_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    workers, shifts = generate_data(seed=42)
    return workers, shifts


@pytest.fixture
def tiny_workers():
    # 10 signups, 8 verified, 4 claimed, 3 completed.
    return pd.DataFrame(
        {
            "worker_id": range(1, 11),
            "signup_date": pd.Timestamp("2025-06-01"),
            "region": ["Bay Area"] * 10,
            "verified": [True] * 8 + [False] * 2,
            "claimed_first_shift": [True] * 4 + [False] * 6,
            "completed_first_shift": [True] * 3 + [False] * 7,
        }
    )


@pytest.fixture
def tiny_shifts():
    return pd.DataFrame(
        {
            "shift_id": range(1, 7),
            "post_date": pd.Timestamp("2025-06-01"),
            "start_date": pd.Timestamp("2025-06-03"),
            "region": ["Bay Area", "Bay Area", "Austin", "Austin", "Chicago", "Chicago"],
            "role": ["Server", "Barista", "Server", "Dishwasher", "Server", "Barista"],
            "lead_time_days": [0, 2, 5, 7, 1, 10],
            "ab_group": ["control", "control", "control", "treatment", "treatment", "treatment"],
            "filled": [True, True, False, True, True, False],
            "no_show": [False, True, False, False, False, False],
        }
    )


# ---------------------------------------------------------------------------
# Determinism / data generation
# ---------------------------------------------------------------------------
def test_generation_is_deterministic():
    w1, s1 = generate_data(seed=42)
    w2, s2 = generate_data(seed=42)
    pd.testing.assert_frame_equal(w1, w2)
    pd.testing.assert_frame_equal(s1, s2)


def test_funnel_logical_consistency(data):
    workers, _ = data
    # A worker can only complete if claimed, only claim if verified.
    assert (workers["claimed_first_shift"] <= workers["verified"]).all()
    assert (workers["completed_first_shift"] <= workers["claimed_first_shift"]).all()


# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------
def test_kpi_summary_values(tiny_workers, tiny_shifts):
    kpi = A.kpi_summary(tiny_workers, tiny_shifts)
    assert kpi["worker_signups"] == 10
    assert kpi["shifts_posted"] == 6
    assert kpi["shifts_filled"] == 4
    assert kpi["fill_rate"] == pytest.approx(4 / 6)
    assert kpi["activation_rate"] == pytest.approx(4 / 10)
    # 1 no-show out of 4 filled.
    assert kpi["no_show_rate"] == pytest.approx(1 / 4)


def test_kpi_rates_in_bounds(data):
    workers, shifts = data
    kpi = A.kpi_summary(workers, shifts)
    for key in ("fill_rate", "activation_rate", "no_show_rate"):
        assert 0.0 <= kpi[key] <= 1.0


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------
def test_build_funnel_counts(tiny_workers):
    funnel = A.build_funnel(tiny_workers)
    assert list(funnel["stage"]) == A.FUNNEL_STAGES
    assert list(funnel["count"]) == [10, 8, 4, 3]
    # Step conversions.
    assert funnel.loc[1, "step_conversion"] == pytest.approx(8 / 10)
    assert funnel.loc[2, "step_conversion"] == pytest.approx(4 / 8)
    # Overall conversion of final stage.
    assert funnel.loc[3, "overall_conversion"] == pytest.approx(3 / 10)
    # Drop-off counts.
    assert funnel.loc[1, "drop_off"] == 2
    assert funnel.loc[2, "drop_off"] == 4


def test_funnel_monotonic_nonincreasing(data):
    workers, _ = data
    funnel = A.build_funnel(workers)
    counts = funnel["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_biggest_dropoff(tiny_workers):
    funnel = A.build_funnel(tiny_workers)
    drop = A.biggest_dropoff(funnel)
    # Verified(8) -> Claimed(4) is the 50% drop, the largest proportional loss.
    assert drop["from_stage"] == "Verified"
    assert drop["to_stage"] == "First Shift Claimed"
    assert drop["drop_off_pct"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fill-rate slices
# ---------------------------------------------------------------------------
def test_fill_rate_by_region(tiny_shifts):
    res = A.fill_rate_by(tiny_shifts, "region")
    # Austin: 1 of 2 filled -> 0.5; Bay Area: 2/2 -> 1.0; Chicago: 1/2 -> 0.5
    rates = dict(zip(res["region"], res["fill_rate"]))
    assert rates["Bay Area"] == pytest.approx(1.0)
    assert rates["Austin"] == pytest.approx(0.5)
    assert rates["Chicago"] == pytest.approx(0.5)
    # Sorted ascending by fill rate (worst first).
    assert res["fill_rate"].is_monotonic_increasing


def test_lead_time_bucket_scalar():
    assert A.lead_time_bucket(0) == "0 days (same-day)"
    assert A.lead_time_bucket(2) == "1-2 days"
    assert A.lead_time_bucket(4) == "3-5 days"
    assert A.lead_time_bucket(9) == "6+ days"


def test_fill_rate_by_lead_time_bucket(tiny_shifts):
    res = A.fill_rate_by(tiny_shifts, "lead_time_bucket")
    assert "lead_time_bucket" in res.columns
    # every posted shift accounted for
    assert res["shifts_posted"].sum() == len(tiny_shifts)


def test_flag_unfilled_segments(tiny_shifts):
    res = A.fill_rate_by(tiny_shifts, "region")
    flagged = A.flag_unfilled_segments(res, target=0.70)
    # Austin & Chicago (0.5) below 0.70; Bay Area (1.0) above.
    below = dict(zip(flagged["region"], flagged["below_target"]))
    assert below["Austin"] is True or below["Austin"]
    assert below["Chicago"]
    assert not below["Bay Area"]
    # Gap to target is positive for below-target segments.
    austin_gap = flagged.loc[flagged["region"] == "Austin", "gap_to_target_pp"].iloc[0]
    assert austin_gap == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------
def test_ztest_known_values():
    # Control 500/1000 = 0.50, Treatment 560/1000 = 0.56.
    res = A.two_proportion_ztest(500, 1000, 560, 1000)
    assert res["p_a"] == pytest.approx(0.50)
    assert res["p_b"] == pytest.approx(0.56)
    assert res["lift_pp"] == pytest.approx(6.0)
    # Hand-computed pooled z: diff 0.06, pooled p 0.53,
    # se = sqrt(0.53*0.47*(2/1000)) = 0.0223261...; z = 2.6874...
    assert res["z"] == pytest.approx(2.6874, abs=1e-3)
    assert res["p_value"] == pytest.approx(0.00720, abs=1e-4)
    assert res["significant"] is True


def test_ztest_ci_contains_diff():
    res = A.two_proportion_ztest(500, 1000, 560, 1000)
    diff = res["p_b"] - res["p_a"]
    assert res["ci_low"] < diff < res["ci_high"]
    # CI is symmetric about the difference (unpooled).
    assert (res["ci_low"] + res["ci_high"]) / 2 == pytest.approx(diff)


def test_ztest_no_effect_not_significant():
    res = A.two_proportion_ztest(500, 1000, 505, 1000)
    assert res["significant"] is False
    assert res["p_value"] > 0.05


def test_ztest_rejects_empty_group():
    with pytest.raises(ValueError):
        A.two_proportion_ztest(0, 0, 10, 100)


def test_ab_fill_rate_designed_effect(data):
    _, shifts = data
    res = A.ab_fill_rate_test(shifts)
    # The designed treatment lift (~+6pp) should be detectable at n~9000.
    assert res["lift_pp"] > 0
    assert res["significant"] is True
    assert res["n_control"] + res["n_treatment"] == len(shifts)
    assert res["ci_low"] > 0  # CI excludes zero
