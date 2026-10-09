"""
analytics.py — Framework-free analytics for the Marketplace Fill-Rate & Worker Funnel dashboard.

Pure data logic only: pandas / numpy / stdlib math. No Streamlit, Plotly, or SciPy
imports here so the functions are unit-testable in isolation and reusable anywhere.

Covered:
  * kpi_summary          — headline marketplace KPIs
  * build_funnel         — worker funnel stages + stage-to-stage & overall conversion
  * fill_rate_by         — fill rate sliced by any dimension (region / role / lead-time bucket)
  * lead_time_bucket     — helper to bucket lead-time days
  * flag_unfilled_segments — flags under-performing segments vs a target
  * two_proportion_ztest — lift, z-stat, two-sided p-value, 95% CI for a difference in proportions
  * ab_fill_rate_test    — applies the z-test to the A/B shift arms
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

FUNNEL_STAGES = ["Signup", "Verified", "First Shift Claimed", "Completed"]


# ---------------------------------------------------------------------------
# Normal distribution helpers (stdlib only — avoids a SciPy dependency)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_sf(x: float) -> float:
    """Standard normal survival function (1 - CDF)."""
    return 1.0 - _norm_cdf(x)


# 97.5th percentile of the standard normal (for a two-sided 95% interval).
Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------
def kpi_summary(workers: pd.DataFrame, shifts: pd.DataFrame) -> dict:
    """Return the headline KPI tiles as a flat dict."""
    n_signups = len(workers)
    n_shifts = len(shifts)
    n_filled = int(shifts["filled"].sum())

    fill_rate = n_filled / n_shifts if n_shifts else 0.0
    # Activation = reached "First Shift Claimed" out of all signups.
    activation_rate = (
        workers["claimed_first_shift"].sum() / n_signups if n_signups else 0.0
    )
    # No-show / cancellation rate is measured over FILLED shifts.
    no_show_rate = shifts["no_show"].sum() / n_filled if n_filled else 0.0

    return {
        "shifts_posted": n_shifts,
        "shifts_filled": n_filled,
        "fill_rate": float(fill_rate),
        "worker_signups": n_signups,
        "activation_rate": float(activation_rate),
        "no_show_rate": float(no_show_rate),
    }


# ---------------------------------------------------------------------------
# Worker funnel
# ---------------------------------------------------------------------------
def build_funnel(workers: pd.DataFrame) -> pd.DataFrame:
    """
    Build the worker funnel.

    Returns a DataFrame with one row per stage and columns:
      stage, count, step_conversion (vs previous stage), overall_conversion (vs Signup),
      drop_off (count lost from previous stage), drop_off_pct.
    """
    counts = [
        len(workers),
        int(workers["verified"].sum()),
        int(workers["claimed_first_shift"].sum()),
        int(workers["completed_first_shift"].sum()),
    ]

    rows = []
    top = counts[0]
    for i, stage in enumerate(FUNNEL_STAGES):
        count = counts[i]
        prev = counts[i - 1] if i > 0 else count
        step_conv = count / prev if prev else 0.0
        overall_conv = count / top if top else 0.0
        drop = (prev - count) if i > 0 else 0
        drop_pct = (drop / prev) if (i > 0 and prev) else 0.0
        rows.append(
            {
                "stage": stage,
                "count": count,
                "step_conversion": step_conv,
                "overall_conversion": overall_conv,
                "drop_off": drop,
                "drop_off_pct": drop_pct,
            }
        )
    return pd.DataFrame(rows)


def biggest_dropoff(funnel: pd.DataFrame) -> dict:
    """Return the funnel transition with the largest proportional drop-off."""
    transitions = funnel.iloc[1:]
    if transitions.empty:
        return {}
    idx = transitions["drop_off_pct"].idxmax()
    row = funnel.loc[idx]
    prev_stage = funnel.loc[idx - 1, "stage"]
    return {
        "from_stage": prev_stage,
        "to_stage": row["stage"],
        "drop_off": int(row["drop_off"]),
        "drop_off_pct": float(row["drop_off_pct"]),
    }


# ---------------------------------------------------------------------------
# Fill-rate slices
# ---------------------------------------------------------------------------
def lead_time_bucket(days) -> "pd.Series | str":
    """Bucket lead-time days into human-readable bands. Accepts scalar or Series."""
    labels = ["0 days (same-day)", "1-2 days", "3-5 days", "6+ days"]
    bins = [-1, 0, 2, 5, np.inf]
    if isinstance(days, pd.Series):
        return pd.cut(days, bins=bins, labels=labels)
    # scalar
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        if lo < days <= hi:
            return lab
    return labels[-1]


def fill_rate_by(shifts: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Fill rate aggregated by a dimension.

    `dimension` is a column name, or the special value "lead_time_bucket" to
    group by the lead-time band. Returns columns:
      <dimension>, shifts_posted, shifts_filled, fill_rate.
    Sorted by fill_rate ascending so the worst (most unfilled) segments are first.
    """
    df = shifts.copy()
    if dimension == "lead_time_bucket":
        df["lead_time_bucket"] = lead_time_bucket(df["lead_time_days"])
        group_col = "lead_time_bucket"
    else:
        group_col = dimension

    grouped = (
        df.groupby(group_col, observed=True)
        .agg(shifts_posted=("filled", "size"), shifts_filled=("filled", "sum"))
        .reset_index()
    )
    grouped["fill_rate"] = grouped["shifts_filled"] / grouped["shifts_posted"]
    return grouped.sort_values("fill_rate", ascending=True).reset_index(drop=True)


def flag_unfilled_segments(
    segment_df: pd.DataFrame, target: float = 0.70, rate_col: str = "fill_rate"
) -> pd.DataFrame:
    """
    Add a boolean `below_target` column and a `gap_to_target` (pp) column flagging
    segments whose fill rate falls below `target`.
    """
    out = segment_df.copy()
    out["below_target"] = out[rate_col] < target
    out["gap_to_target_pp"] = (target - out[rate_col]) * 100
    return out


# ---------------------------------------------------------------------------
# A/B test — two-proportion z-test
# ---------------------------------------------------------------------------
def two_proportion_ztest(
    success_a: int, n_a: int, success_b: int, n_b: int
) -> dict:
    """
    Two-sided two-proportion z-test comparing group B (treatment) vs group A (control).

    Returns:
      p_a, p_b            — observed proportions
      lift_pp             — (p_b - p_a) in percentage points
      relative_lift       — (p_b - p_a) / p_a
      z                   — z-statistic (pooled SE)
      p_value             — two-sided p-value
      ci_low, ci_high     — 95% CI for (p_b - p_a) in PROPORTION units (unpooled SE)
      significant         — p_value < 0.05
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Both groups must have at least one observation.")

    p_a = success_a / n_a
    p_b = success_b / n_b
    diff = p_b - p_a

    # Pooled proportion for the hypothesis test.
    p_pool = (success_a + success_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * _norm_sf(abs(z))

    # Unpooled SE for the confidence interval of the difference.
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = diff - Z_95 * se_unpooled
    ci_high = diff + Z_95 * se_unpooled

    return {
        "p_a": p_a,
        "p_b": p_b,
        "lift_pp": diff * 100,
        "relative_lift": (diff / p_a) if p_a > 0 else float("nan"),
        "z": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p_value < 0.05,
    }


def ab_fill_rate_test(
    shifts: pd.DataFrame,
    group_col: str = "ab_group",
    control_label: str = "control",
    treatment_label: str = "treatment",
    outcome_col: str = "filled",
) -> dict:
    """Run the two-proportion z-test on fill rate for control vs treatment shift arms."""
    control = shifts[shifts[group_col] == control_label]
    treatment = shifts[shifts[group_col] == treatment_label]

    n_c, s_c = len(control), int(control[outcome_col].sum())
    n_t, s_t = len(treatment), int(treatment[outcome_col].sum())

    result = two_proportion_ztest(s_c, n_c, s_t, n_t)
    result.update(
        {
            "n_control": n_c,
            "n_treatment": n_t,
            "filled_control": s_c,
            "filled_treatment": s_t,
        }
    )
    return result
