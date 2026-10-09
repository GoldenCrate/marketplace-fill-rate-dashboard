"""
app.py — Marketplace Fill-Rate & Worker Funnel dashboard (Streamlit).

A two-sided gig-marketplace analytics demo:
  1. KPI tiles            — headline marketplace health
  2. Worker funnel        — Signup -> Verified -> First Shift Claimed -> Completed
  3. Shift fill-rate view — fill rate by region / role / lead time, flagging weak segments
  4. A/B experiment       — matching-algorithm tweak, two-proportion z-test
  5. "So what" notes      — a recommendation on every view

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data": takeaway-style
titles, gray for context + a single accent colour for the point, direct labels,
and minimal chart chrome.

  ALL DATA IS SYNTHETIC (fixed-seed, generated in data_gen.py). See README.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from data_gen import SEED, generate_data

# ---------------------------------------------------------------------------
# Palette (Knaflic: mostly gray, one accent; a muted red only for "needs action")
# ---------------------------------------------------------------------------
GRAY = "#B9C0C9"
GRAY_DARK = "#5A6572"
ACCENT = "#1F6FEB"      # blue — "this is the point"
GOOD = "#2E7D5B"        # muted green
WARN = "#C0442E"        # muted red — flagged / needs action
INK = "#222933"

st.set_page_config(
    page_title="Marketplace Fill-Rate & Worker Funnel",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    return generate_data(seed=SEED)


workers, shifts = load_data()


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def so_what(text: str) -> None:
    """Render a consistent 'So what / recommendation' callout."""
    st.markdown(
        f"""
        <div style="border-left:4px solid {ACCENT};background:#F2F6FD;
                    padding:14px 18px;border-radius:4px;margin-top:8px;">
          <span style="font-weight:700;color:{ACCENT};">So what →</span>
          <span style="color:{INK};"> {text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply a minimal Knaflic-style layout: takeaway title, no clutter."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=INK)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Helvetica, Arial, sans-serif", color=GRAY_DARK, size=13),
        margin=dict(l=20, r=20, t=60, b=40),
        height=height,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Marketplace Analytics")
st.sidebar.caption("Fill-Rate & Worker Funnel")
view = st.sidebar.radio(
    "View",
    [
        "KPI Overview",
        "Worker Funnel",
        "Shift Fill Rate",
        "A/B: Matching Algorithm",
    ],
)
st.sidebar.markdown("---")
region_filter = st.sidebar.multiselect(
    "Filter regions (fill-rate & KPI views)",
    options=sorted(shifts["region"].unique()),
    default=sorted(shifts["region"].unique()),
)
st.sidebar.markdown("---")
st.sidebar.warning("All data is SYNTHETIC (fixed seed). Portfolio demo only.")
st.sidebar.caption("Leo Chan · github.com/GoldenCrate")

# Apply region filter to a working copy (A/B view intentionally uses full data).
f_shifts = shifts[shifts["region"].isin(region_filter)] if region_filter else shifts
f_workers = workers[workers["region"].isin(region_filter)] if region_filter else workers


# ===========================================================================
# VIEW 1 — KPI OVERVIEW
# ===========================================================================
def view_kpis():
    st.title("Marketplace Health at a Glance")
    st.caption("Synthetic two-sided gig marketplace · 90-day window")

    kpi = A.kpi_summary(f_workers, f_shifts)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shifts Posted", f"{kpi['shifts_posted']:,}")
    c2.metric("Fill Rate", pct(kpi["fill_rate"]),
              help="Share of posted shifts that got filled")
    c3.metric("Worker Signups", f"{kpi['worker_signups']:,}")
    c4.metric("Activation Rate", pct(kpi["activation_rate"]),
              help="Signups that claimed a first shift")
    c5.metric("No-Show / Cancel Rate", pct(kpi["no_show_rate"]),
              help="Of filled shifts, share ending in a no-show or cancellation")

    st.markdown("---")

    left, right = st.columns(2)

    # Fill rate trend over time (weekly) — context in gray, latest point in accent.
    with left:
        tmp = f_shifts.copy()
        tmp["week"] = tmp["post_date"].dt.to_period("W").dt.start_time
        weekly = tmp.groupby("week").agg(
            posted=("filled", "size"), filled=("filled", "sum")
        )
        weekly["fill_rate"] = weekly["filled"] / weekly["posted"]
        weekly = weekly.reset_index()

        colors = [GRAY] * len(weekly)
        if colors:
            colors[-1] = ACCENT
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=weekly["week"],
                y=weekly["fill_rate"],
                mode="lines+markers",
                line=dict(color=GRAY, width=2),
                marker=dict(color=colors, size=7),
                hovertemplate="%{x|%b %d}<br>Fill rate %{y:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        clean_layout(fig, "Weekly fill rate holds in the low-to-mid 70s", height=360)
        st.plotly_chart(fig, width="stretch")

    # Signup -> activation mini bar
    with right:
        n = len(f_workers)
        activated = int(f_workers["claimed_first_shift"].sum())
        completed = int(f_workers["completed_first_shift"].sum())
        stages = ["Signed up", "Claimed 1st shift", "Completed 1st shift"]
        vals = [n, activated, completed]
        bar_colors = [GRAY, ACCENT, GRAY_DARK]
        fig = go.Figure(
            go.Bar(
                x=stages, y=vals, marker_color=bar_colors,
                text=[f"{v:,}" for v in vals], textposition="outside",
            )
        )
        clean_layout(fig, "Under half of signups reach their first shift", height=360)
        fig.update_yaxes(range=[0, n * 1.15])
        st.plotly_chart(fig, width="stretch")

    so_what(
        f"Fill rate sits near {pct(kpi['fill_rate'], 0)} and activation at "
        f"{pct(kpi['activation_rate'], 0)} — both demand (unfilled shifts) and supply "
        f"(signups that never work) leak value. The funnel and fill-rate views below "
        f"isolate <b>where</b> to intervene first."
    )


# ===========================================================================
# VIEW 2 — WORKER FUNNEL
# ===========================================================================
def view_funnel():
    st.title("Where Workers Fall Out of the Funnel")
    st.caption("Signup → Verified → First Shift Claimed → Completed (synthetic)")

    funnel = A.build_funnel(f_workers)
    drop = A.biggest_dropoff(funnel)

    # Highlight the worst transition's destination stage in accent; rest gray.
    worst_stage = drop.get("to_stage")
    bar_colors = [ACCENT if s == worst_stage else GRAY for s in funnel["stage"]]

    fig = go.Figure(
        go.Bar(
            y=funnel["stage"][::-1],
            x=funnel["count"][::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[
                f"{c:,}  ({o:.0%})"
                for c, o in zip(funnel["count"][::-1], funnel["overall_conversion"][::-1])
            ],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:,} workers<extra></extra>",
        )
    )
    clean_layout(
        fig,
        f"Biggest leak: {drop['from_stage']} → {drop['to_stage']} "
        f"loses {drop['drop_off_pct']:.0%}",
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

    # Step-conversion detail table.
    tbl = funnel.copy()
    tbl["step_conversion"] = tbl["step_conversion"].map(lambda v: pct(v))
    tbl["overall_conversion"] = tbl["overall_conversion"].map(lambda v: pct(v))
    tbl["drop_off_pct"] = tbl["drop_off_pct"].map(lambda v: pct(v) if v else "—")
    tbl = tbl.rename(
        columns={
            "stage": "Stage",
            "count": "Workers",
            "step_conversion": "Step conversion",
            "overall_conversion": "Overall conversion",
            "drop_off": "Dropped",
            "drop_off_pct": "Step drop-off",
        }
    )
    st.dataframe(tbl, width="stretch", hide_index=True)

    so_what(
        f"The {drop['from_stage']} → {drop['to_stage']} step loses "
        f"<b>{drop['drop_off']:,} workers ({drop['drop_off_pct']:.0%})</b> — the single "
        f"largest leak. Verified workers who never claim a first shift are the cheapest "
        f"supply to recover: target them with first-shift nudges, nearby-shift alerts, "
        f"and a lighter claim flow before spending on new acquisition."
    )


# ===========================================================================
# VIEW 3 — SHIFT FILL RATE
# ===========================================================================
def view_fill_rate():
    st.title("Which Shifts Go Unfilled")
    st.caption("Fill rate by segment, flagging anything below target (synthetic)")

    target = st.slider(
        "Fill-rate target", min_value=0.50, max_value=0.95, value=0.70, step=0.01,
        format="%.0f%%",
    )
    dim_label = st.radio(
        "Break down by", ["Region", "Role", "Lead time"], horizontal=True
    )
    dim_map = {"Region": "region", "Role": "role", "Lead time": "lead_time_bucket"}
    dimension = dim_map[dim_label]

    seg = A.fill_rate_by(f_shifts, dimension)
    seg = A.flag_unfilled_segments(seg, target=target)
    cat_col = seg.columns[0]

    # Bars below target in muted red, at/above in gray; sorted worst-first already.
    bar_colors = [WARN if below else GRAY for below in seg["below_target"]]

    fig = go.Figure(
        go.Bar(
            x=seg[cat_col].astype(str),
            y=seg["fill_rate"],
            marker_color=bar_colors,
            text=[pct(v, 0) for v in seg["fill_rate"]],
            textposition="outside",
            hovertemplate="%{x}<br>Fill rate %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=target, line_dash="dash", line_color=INK,
        annotation_text=f"Target {pct(target, 0)}", annotation_position="top left",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    n_below = int(seg["below_target"].sum())
    clean_layout(
        fig,
        f"{n_below} of {len(seg)} {dim_label.lower()} segments sit below the "
        f"{pct(target, 0)} target",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    flagged = seg[seg["below_target"]].copy()
    if not flagged.empty:
        show = flagged[[cat_col, "shifts_posted", "shifts_filled", "fill_rate", "gap_to_target_pp"]].copy()
        show["fill_rate"] = show["fill_rate"].map(lambda v: pct(v))
        show["gap_to_target_pp"] = show["gap_to_target_pp"].map(lambda v: f"{v:.1f} pp")
        show = show.rename(
            columns={
                cat_col: dim_label,
                "shifts_posted": "Posted",
                "shifts_filled": "Filled",
                "fill_rate": "Fill rate",
                "gap_to_target_pp": "Gap to target",
            }
        )
        st.markdown("**Flagged segments (below target):**")
        st.dataframe(show, width="stretch", hide_index=True)

    worst = seg.iloc[0]
    so_what(
        f"<b>{worst[cat_col]}</b> is the weakest {dim_label.lower()} segment at "
        f"{pct(worst['fill_rate'], 0)} fill "
        f"({int(worst['shifts_posted']):,} shifts posted). "
        + (
            "Lead time is the clearest lever: same-day and 1–2 day posts fill worst, so "
            "nudging businesses to post earlier — plus boosted pay or wider worker radius "
            "on the flagged segments — should lift fill fastest."
            if dimension == "lead_time_bucket"
            else f"Concentrate matching incentives and supply recruitment on the flagged "
            f"{dim_label.lower()} segments rather than spreading spend evenly."
        )
    )


# ===========================================================================
# VIEW 4 — A/B EXPERIMENT
# ===========================================================================
def view_ab():
    st.title("Does the New Matching Algorithm Lift Fill Rate?")
    st.caption(
        "A/B test: control (current matching) vs treatment (algorithm tweak). "
        "Two-proportion z-test on shift fill rate. Synthetic experiment."
    )

    show_treatment = st.toggle("Show treatment arm result", value=True)

    res = A.ab_fill_rate_test(shifts)  # full data — A/B is not region-filtered

    control_rate = res["p_a"]
    treatment_rate = res["p_b"] if show_treatment else res["p_a"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Control fill rate", pct(control_rate),
        help=f"{res['filled_control']:,} of {res['n_control']:,} shifts",
    )
    c2.metric(
        "Treatment fill rate",
        pct(treatment_rate) if show_treatment else "—",
        delta=f"{res['lift_pp']:+.1f} pp" if show_treatment else None,
        help=f"{res['filled_treatment']:,} of {res['n_treatment']:,} shifts",
    )
    c3.metric(
        "Two-sided p-value",
        f"{res['p_value']:.4f}" if show_treatment else "—",
        help="From a two-proportion z-test (pooled SE)",
    )

    if show_treatment:
        # Two bars: control gray, treatment accent. Annotate the lift.
        fig = go.Figure(
            go.Bar(
                x=["Control", "Treatment"],
                y=[res["p_a"], res["p_b"]],
                marker_color=[GRAY, ACCENT],
                text=[pct(res["p_a"]), pct(res["p_b"])],
                textposition="outside",
                width=0.5,
            )
        )
        fig.update_yaxes(tickformat=".0%", range=[0, max(res["p_a"], res["p_b"]) * 1.25])
        verdict = "statistically significant" if res["significant"] else "not significant"
        clean_layout(
            fig,
            f"Treatment lifts fill rate {res['lift_pp']:+.1f} pp "
            f"({res['relative_lift'] * 100:+.1f}% relative) — {verdict}",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Statistical detail (two-proportion z-test)**")
        detail = pd.DataFrame(
            {
                "Metric": [
                    "Absolute lift",
                    "Relative lift",
                    "z-statistic",
                    "p-value (two-sided)",
                    "95% CI for lift",
                    "Significant at α=0.05?",
                ],
                "Value": [
                    f"{res['lift_pp']:+.2f} pp",
                    f"{res['relative_lift'] * 100:+.2f}%",
                    f"{res['z']:.3f}",
                    f"{res['p_value']:.4f}",
                    f"[{res['ci_low'] * 100:+.2f}, {res['ci_high'] * 100:+.2f}] pp",
                    "Yes" if res["significant"] else "No",
                ],
            }
        )
        st.dataframe(detail, width="stretch", hide_index=True)

        if res["significant"] and res["ci_low"] > 0:
            so_what(
                f"The tweak raises fill rate by <b>{res['lift_pp']:.1f} pp</b> "
                f"(95% CI {res['ci_low'] * 100:.1f}–{res['ci_high'] * 100:.1f} pp, "
                f"p={res['p_value']:.4f}); the interval excludes zero, so the effect is real, "
                f"not noise. <b>Recommendation: ship the new matching algorithm</b> and "
                f"monitor fill rate and no-show rate during rollout. On "
                f"{res['n_control'] + res['n_treatment']:,} shifts a "
                f"{res['lift_pp']:.1f} pp lift is a meaningful supply-utilisation gain."
            )
        else:
            so_what(
                "The observed lift is not statistically distinguishable from zero at "
                "α=0.05. Recommendation: hold the rollout and gather more data or redesign "
                "the tweak before committing."
            )
    else:
        so_what(
            "Toggle on the treatment arm to compare fill rate and read the z-test verdict."
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if view == "KPI Overview":
    view_kpis()
elif view == "Worker Funnel":
    view_funnel()
elif view == "Shift Fill Rate":
    view_fill_rate()
elif view == "A/B: Matching Algorithm":
    view_ab()

st.markdown("---")
st.caption(
    "Synthetic data (fixed seed) · built with Streamlit, pandas & Plotly · "
    "charts follow Storytelling with Data (Knaflic) · Leo Chan — github.com/GoldenCrate"
)
