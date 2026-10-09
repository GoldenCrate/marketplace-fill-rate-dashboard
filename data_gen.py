"""
data_gen.py — SYNTHETIC data generator for the Marketplace Fill-Rate & Worker Funnel dashboard.

================================================================================
  !!  ALL DATA PRODUCED HERE IS 100% SYNTHETIC AND FICTIONAL.  !!
  It is generated from a fixed random seed for a portfolio demo and does NOT
  represent Instawork, any real marketplace, or any real workers or businesses.
================================================================================

Two-sided gig-marketplace model:
  * WORKERS move through a funnel: Signup -> Verified -> First Shift Claimed -> Completed.
  * SHIFTS are posted by businesses and either get FILLED or go unfilled, and
    filled shifts can end in a no-show / cancellation.
  * An A/B experiment on a matching-algorithm tweak is baked into the shift data
    with a DESIGNED, statistically detectable lift on fill rate (treatment > control).

Run `python data_gen.py` to print a summary of the generated frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

# ---------------------------------------------------------------------------
# Marketplace parameters (synthetic, chosen for a realistic-looking demo)
# ---------------------------------------------------------------------------
N_WORKERS = 6000
N_SHIFTS = 9000

REGIONS = ["Bay Area", "Los Angeles", "Seattle", "Austin", "Chicago"]
# Relative share of volume per region (sums to 1 after normalisation).
REGION_WEIGHTS = np.array([0.30, 0.25, 0.15, 0.15, 0.15])

ROLES = ["Server", "Prep Cook", "Warehouse Associate", "Event Staff", "Barista", "Dishwasher"]
ROLE_WEIGHTS = np.array([0.22, 0.18, 0.20, 0.15, 0.13, 0.12])

# Worker funnel conversion rates (per stage, conditional on the previous stage).
P_VERIFY = 0.78          # Signup -> Verified
P_CLAIM_GIVEN_VERIFY = 0.55   # Verified -> First Shift Claimed
P_COMPLETE_GIVEN_CLAIM = 0.86  # First Shift Claimed -> Completed

# Per-region nudge on verification (some markets onboard better than others).
REGION_VERIFY_LIFT = {
    "Bay Area": 0.03,
    "Los Angeles": 0.0,
    "Seattle": 0.02,
    "Austin": -0.02,
    "Chicago": -0.03,
}

# Base fill probability by role (before region, lead-time and A/B effects).
ROLE_FILL_BASE = {
    "Server": 0.74,
    "Prep Cook": 0.70,
    "Warehouse Associate": 0.80,
    "Event Staff": 0.66,
    "Barista": 0.72,
    "Dishwasher": 0.62,
}

# Per-region additive effect on fill probability.
REGION_FILL_LIFT = {
    "Bay Area": 0.04,
    "Los Angeles": 0.01,
    "Seattle": 0.00,
    "Austin": -0.02,
    "Chicago": -0.03,
}

# A/B experiment: matching-algorithm tweak applied to the TREATMENT arm only.
# This is the DESIGNED effect the A/B view is built to detect.
AB_TREATMENT_FILL_LIFT = 0.06   # +6 percentage points (additive) for treatment
AB_SPLIT = 0.50                 # 50/50 assignment

NO_SHOW_RATE = 0.07             # of FILLED shifts, share that end in no-show/cancel


def _bounded(p: np.ndarray) -> np.ndarray:
    """Clip probabilities into a safe (0, 1) range."""
    return np.clip(p, 0.01, 0.99)


def generate_workers(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the synthetic WORKER funnel table."""
    worker_id = np.arange(1, N_WORKERS + 1)
    signup_day = rng.integers(0, 90, size=N_WORKERS)  # day index over a 90-day window
    signup_date = pd.Timestamp("2025-06-01") + pd.to_timedelta(signup_day, unit="D")
    region = rng.choice(REGIONS, size=N_WORKERS, p=REGION_WEIGHTS / REGION_WEIGHTS.sum())

    # Stage 1 -> 2: Verified
    p_verify = _bounded(
        np.array([P_VERIFY + REGION_VERIFY_LIFT[r] for r in region])
    )
    verified = rng.random(N_WORKERS) < p_verify

    # Stage 2 -> 3: First Shift Claimed (only possible if verified)
    claim_draw = rng.random(N_WORKERS) < P_CLAIM_GIVEN_VERIFY
    claimed_first_shift = verified & claim_draw

    # Stage 3 -> 4: Completed first shift (only possible if claimed)
    complete_draw = rng.random(N_WORKERS) < P_COMPLETE_GIVEN_CLAIM
    completed_first_shift = claimed_first_shift & complete_draw

    return pd.DataFrame(
        {
            "worker_id": worker_id,
            "signup_date": signup_date,
            "region": region,
            "verified": verified,
            "claimed_first_shift": claimed_first_shift,
            "completed_first_shift": completed_first_shift,
        }
    )


def generate_shifts(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the synthetic SHIFT / fill-rate table with the A/B experiment baked in."""
    shift_id = np.arange(1, N_SHIFTS + 1)
    region = rng.choice(REGIONS, size=N_SHIFTS, p=REGION_WEIGHTS / REGION_WEIGHTS.sum())
    role = rng.choice(ROLES, size=N_SHIFTS, p=ROLE_WEIGHTS / ROLE_WEIGHTS.sum())

    post_day = rng.integers(0, 90, size=N_SHIFTS)
    post_date = pd.Timestamp("2025-06-01") + pd.to_timedelta(post_day, unit="D")

    # Lead time = days between posting and shift start. Short lead time = harder to fill.
    lead_time_days = rng.integers(0, 15, size=N_SHIFTS)
    start_date = post_date + pd.to_timedelta(lead_time_days, unit="D")

    # A/B assignment on the matching algorithm.
    ab_group = np.where(rng.random(N_SHIFTS) < AB_SPLIT, "treatment", "control")

    # Build per-shift fill probability.
    base = np.array([ROLE_FILL_BASE[r] for r in role])
    region_eff = np.array([REGION_FILL_LIFT[r] for r in region])
    # Lead-time effect: more lead time -> higher fill (diminishing). 0 days is worst.
    lead_eff = 0.015 * np.minimum(lead_time_days, 10) - 0.08
    ab_eff = np.where(ab_group == "treatment", AB_TREATMENT_FILL_LIFT, 0.0)

    p_fill = _bounded(base + region_eff + lead_eff + ab_eff)
    filled = rng.random(N_SHIFTS) < p_fill

    # No-show / cancellation only applies to FILLED shifts.
    no_show = filled & (rng.random(N_SHIFTS) < NO_SHOW_RATE)

    return pd.DataFrame(
        {
            "shift_id": shift_id,
            "post_date": post_date,
            "start_date": start_date,
            "region": region,
            "role": role,
            "lead_time_days": lead_time_days,
            "ab_group": ab_group,
            "filled": filled,
            "no_show": no_show,
        }
    )


def generate_data(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (workers_df, shifts_df). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    workers = generate_workers(rng)
    shifts = generate_shifts(rng)
    return workers, shifts


if __name__ == "__main__":
    workers, shifts = generate_data()
    print("=" * 70)
    print("SYNTHETIC DATA SUMMARY (fixed seed =", SEED, ")")
    print("=" * 70)
    print(f"\nWorkers: {len(workers):,} rows")
    print(workers.head())
    print("\nFunnel counts:")
    print("  Signup              :", len(workers))
    print("  Verified            :", int(workers.verified.sum()))
    print("  First Shift Claimed :", int(workers.claimed_first_shift.sum()))
    print("  Completed           :", int(workers.completed_first_shift.sum()))

    print(f"\nShifts: {len(shifts):,} rows")
    print(shifts.head())
    fill = shifts.filled.mean()
    print(f"\nOverall fill rate: {fill:.1%}")
    ab = shifts.groupby("ab_group").filled.mean()
    print("Fill rate by A/B group:")
    print(ab.to_string())
    print(f"Observed treatment - control lift: {(ab.get('treatment', 0) - ab.get('control', 0)) * 100:.2f} pp")
