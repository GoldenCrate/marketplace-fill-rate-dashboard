# Marketplace Fill-Rate & Worker Funnel Dashboard

A Product-Operations analytics dashboard for a **two-sided gig-work marketplace**
(hourly workers ↔ businesses). It tracks marketplace health, diagnoses where
workers fall out of the onboarding funnel, flags which shift segments go unfilled,
and evaluates a matching-algorithm A/B test with a proper two-proportion z-test.

Built as a portfolio project by **Leo Chan** for a Product Operations Analyst
application. It mirrors the day-to-day questions a marketplace ops team asks:
*Are we filling shifts? Where are we leaking supply? Did the change we shipped work?*

> ⚠️ **Synthetic-data disclaimer**
> **All data in this project is 100% synthetic and fictional**, generated from a
> fixed random seed in [`data_gen.py`](data_gen.py). It does **not** represent
> Instawork, any real marketplace, or any real workers, businesses, or shifts.
> The A/B "result" reflects a deliberately designed effect, not a real experiment.

---

## Views

1. **KPI Overview** — headline tiles: shifts posted, fill rate %, worker signups,
   activation rate, and no-show / cancellation rate, plus a weekly fill-rate trend
   and a signup → first-shift bar.
2. **Worker Funnel** — Signup → Verified → First Shift Claimed → Completed, with
   step-by-step conversion, drop-off counts, and automatic detection of the
   biggest leak.
3. **Shift Fill Rate** — fill rate sliced by **region**, **role**, or **lead time**,
   with an adjustable target line and automatic flagging of under-target segments.
4. **A/B: Matching Algorithm** — toggle a matching-algorithm tweak (treatment) vs
   the current algorithm (control). Reports absolute & relative lift, z-statistic,
   two-sided p-value, and a 95% confidence interval for the difference in fill rate.
5. **"So what →" recommendation** on every view, translating the numbers into an
   ops decision.

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: takeaway-style
chart titles, gray for context with a single accent colour for the point, direct
data labels, and minimal chart chrome.

---

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Multi-view Streamlit dashboard |
| `analytics.py` | Framework-free analytics (KPIs, funnel, fill-rate slices, z-test) — pandas/numpy/stdlib only |
| `data_gen.py` | Synthetic worker & shift data generator (fixed seed, designed A/B effect) |
| `test_analytics.py` | pytest suite for the analytics layer |
| `requirements.txt` | Dependencies |

`analytics.py` imports **no** Streamlit, Plotly, or SciPy — the statistics are
implemented with the standard library (`math.erf` for the normal CDF), so the
logic is portable and unit-testable on its own.

---

## Run it locally

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. preview the synthetic data
python data_gen.py

# 4. run the tests
pytest -q

# 5. launch the dashboard
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

---

## Methodology notes

- **Fill rate** = filled shifts ÷ posted shifts.
- **Activation rate** = signups who claimed a first shift ÷ total signups.
- **No-show / cancellation rate** = measured over *filled* shifts.
- **A/B test** uses a two-proportion z-test: a **pooled** standard error for the
  z-statistic / p-value, and an **unpooled** standard error for the 95% CI of the
  difference. The synthetic generator bakes in a ~+6 percentage-point treatment
  lift so the test detects a real, significant effect.

---

## Author

**Leo Chan** — [github.com/GoldenCrate](https://github.com/GoldenCrate)

Portfolio project. Synthetic data only.
