"""
Aegis-v2 Mission Console (Streamlit).

Interactive view over campaign results written by experiments/run_campaign.py.

    streamlit run dashboard/app.py
"""

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aegis.reporting import STRATEGIES, SUMMARY_KEYS, list_campaigns, load_seed_runs, pareto_points  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
LABELS = [label for _, label, _, _ in STRATEGIES]
COLORS = [color for _, _, color, _ in STRATEGIES]

st.set_page_config(page_title="Aegis-v2 Mission Console", layout="wide")
st.title("Aegis-v2 Mission Console")
st.caption(
    "ResNet-18 EuroSAT classifier in a 600 km sun-synchronous LEO orbit. Five replicas receive the "
    "identical radiation fault stream; each applies a different mitigation strategy."
)

campaigns = list_campaigns(RESULTS_DIR)
if not campaigns:
    st.warning("No campaign results found. Run `python experiments/run_campaign.py` first.")
    st.stop()

with st.sidebar:
    campaign = st.selectbox("Campaign", campaigns, index=campaigns.index("main") if "main" in campaigns else 0)
    runs = load_seed_runs(os.path.join(RESULTS_DIR, campaign))
    seed_options = ["Mean of seeds"] + [f"Seed {s}" for s in sorted(runs)]
    seed_choice = st.selectbox("Seed", seed_options)

summaries = [r["summary"] for r in runs.values()]


def telemetry_frame(seed_choice: str) -> pd.DataFrame:
    if seed_choice == "Mean of seeds":
        frames = [pd.DataFrame(r["telemetry"]) for r in runs.values()]
        numeric = pd.concat(frames).groupby("step_index").mean(numeric_only=True).reset_index()
        numeric["in_saa"] = frames[0]["in_saa"].values
        numeric["aegis_restored_blocks"] = ""
        return numeric
    df = pd.DataFrame(runs[int(seed_choice.split()[-1])]["telemetry"])
    df["aegis_restored_blocks"] = df["aegis_restored_blocks"].apply(", ".join)
    return df


df = telemetry_frame(seed_choice)
days = float(summaries[0]["total_simulated_days"]) or 1.0

# ---- Headline numbers -------------------------------------------------------------------------
mean_of = lambda key: sum(s["mean_accuracy"][key] for s in summaries) / len(summaries)  # noqa: E731
bytes_of = lambda key: sum(s["bytes_transferred"][key] for s in summaries) / len(summaries) / days / 1e6  # noqa: E731
cols = st.columns(4)
cols[0].metric("Clean accuracy", f"{summaries[0]['clean_accuracy'] * 100:.1f}%")
cols[1].metric("Aegis-v2 mean accuracy", f"{mean_of('aegis_v2') * 100:.1f}%",
               f"{(mean_of('aegis_v2') - mean_of('periodic_scrubbing')) * 100:+.1f} pts vs 12 h scrub")
cols[2].metric("Aegis-v2 traffic", f"{bytes_of('aegis_v2'):.0f} MB/day",
               f"12 h scrub: {bytes_of('periodic_scrubbing'):.0f} MB/day", delta_color="off")
cols[3].metric("Bit flips accumulated", f"{sum(s['total_faults_accumulated'] for s in summaries) / len(summaries):.0f}",
               f"{sum(s['exponent_msb_flips_total'] for s in summaries) / len(summaries):.0f} exponent-MSB",
               delta_color="off")
st.caption(
    f"{len(runs)} seed(s) · {days:g} simulated days · fault field: {summaries[0]['fault_field']} · "
    "accuracy is time-averaged over every step."
)

tab_acc, tab_pareto, tab_faults, tab_table = st.tabs(
    ["Accuracy over time", "Accuracy vs bandwidth", "Faults & repairs", "Data table"]
)

# ---- Accuracy over time -----------------------------------------------------------------------
with tab_acc:
    long = df.melt(id_vars=["elapsed_days"], value_vars=[a for a, *_ in STRATEGIES],
                   var_name="attr", value_name="accuracy")
    long["strategy"] = long["attr"].map({a: label for a, label, *_ in STRATEGIES})
    color = alt.Color("strategy:N", scale=alt.Scale(domain=LABELS, range=COLORS),
                      legend=alt.Legend(orient="top", title=None))
    hover = alt.selection_point(fields=["elapsed_days"], nearest=True, on="pointerover", empty=False)

    saa = alt.Chart(df[df["in_saa"].astype(bool)]).mark_rect(color="#888", opacity=0.12).encode(
        x=alt.X("start:Q"), x2="elapsed_days:Q"
    ).transform_calculate(start=f"datum.elapsed_days - {1 / 24}")
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("elapsed_days:Q", title="Mission day"),
        y=alt.Y("accuracy:Q", title="Top-1 accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=color,
    )
    rule = alt.Chart(long).mark_rule(color="#999").encode(
        x="elapsed_days:Q",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("elapsed_days:Q", title="Day", format=".2f")],
    ).add_params(hover)
    points = lines.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["strategy:N", alt.Tooltip("accuracy:Q", format=".1%"),
                 alt.Tooltip("elapsed_days:Q", title="Day", format=".2f")],
    )
    st.altair_chart((saa + lines + rule + points).properties(height=380), width="stretch")
    st.caption("Shaded columns: steps that include a South Atlantic Anomaly pass.")

# ---- Pareto -----------------------------------------------------------------------------------
with tab_pareto:
    pdf = pd.DataFrame(pareto_points(runs))
    pdf["label"] = pdf["strategy"]
    family_color = alt.Color(
        "family:N",
        scale=alt.Scale(domain=["aegis", "scrub", "uplink", "baseline"],
                        range=[COLORS[0], COLORS[1], COLORS[5], "#8a8984"]),
        legend=alt.Legend(orient="top", title=None,
                          labelExpr="{'aegis':'Aegis-v2','scrub':'Periodic scrubbing','uplink':'Ground uplink','baseline':'No-traffic baselines'}[datum.label]"),
    )
    base = alt.Chart(pdf).encode(
        x=alt.X("mb_per_day:Q", title="Golden-store traffic (MB / day, symlog)",
                scale=alt.Scale(type="symlog", constant=10)),
        y=alt.Y("mean_accuracy:Q", title="Time-averaged accuracy", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False)),
    )
    scrub_line = base.transform_filter("datum.family == 'scrub'").mark_line(color=COLORS[1], strokeWidth=2)
    dots = base.mark_circle(size=140, stroke="white", strokeWidth=2, opacity=1).encode(
        color=family_color,
        tooltip=["strategy:N", alt.Tooltip("mean_accuracy:Q", format=".1%"),
                 alt.Tooltip("mb_per_day:Q", title="MB/day", format=".1f")],
    )
    text = base.mark_text(align="left", dx=9, dy=-8, fontSize=11, color="#52514e").encode(text="label:N")
    st.altair_chart((scrub_line + dots + text).properties(height=420), width="stretch")
    st.markdown(
        "Each scrubbing interval trades bandwidth for accuracy. A strategy dominates if it sits above and "
        "to the left of the scrubbing curve. Ranger, BLINK, and unmitigated move no golden-store bytes "
        "(they never repair memory), so they sit at 0 MB/day."
    )

# ---- Faults & repairs -------------------------------------------------------------------------
with tab_faults:
    left, right = st.columns(2)
    flips = df.melt(id_vars=["elapsed_days"], value_vars=["bit_flips_step", "exponent_msb_flips_step"],
                    var_name="kind", value_name="flips")
    flips["kind"] = flips["kind"].map({"bit_flips_step": "All bits", "exponent_msb_flips_step": "Exponent MSB"})
    left.altair_chart(
        alt.Chart(flips).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("elapsed_days:Q", title="Mission day"),
            y=alt.Y("flips:Q", title="Bit flips per step"),
            color=alt.Color("kind:N", scale=alt.Scale(range=[COLORS[0], COLORS[1]]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["kind:N", alt.Tooltip("flips:Q", format=".1f"),
                     alt.Tooltip("elapsed_days:Q", title="Day", format=".2f")],
        ).properties(height=320, title="Radiation fault stream"),
        width="stretch",
    )
    energy = df.melt(id_vars=["elapsed_days"], value_vars=["bytes_aegis", "bytes_scrub"],
                     var_name="who", value_name="bytes")
    energy["MB"] = energy["bytes"] / 1e6
    energy["who"] = energy["who"].map({"bytes_aegis": LABELS[0], "bytes_scrub": LABELS[1]})
    right.altair_chart(
        alt.Chart(energy).mark_line(strokeWidth=2, interpolate="step-after").encode(
            x=alt.X("elapsed_days:Q", title="Mission day"),
            y=alt.Y("MB:Q", title="Cumulative golden-store traffic (MB)"),
            color=alt.Color("who:N", scale=alt.Scale(domain=LABELS[:2], range=COLORS[:2]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["who:N", alt.Tooltip("MB:Q", format=".1f"), alt.Tooltip("elapsed_days:Q", format=".2f")],
        ).properties(height=320, title="Cumulative repair traffic"),
        width="stretch",
    )
    if seed_choice != "Mean of seeds":
        repairs = df[df["aegis_restored_blocks"] != ""][
            ["step_index", "elapsed_days", "in_saa", "bit_flips_step", "exponent_msb_flips_step",
             "aegis_restored_blocks", "acc_unmitigated", "acc_aegis"]
        ]
        st.subheader(f"Aegis-v2 repair log ({len(repairs)} events)")
        st.dataframe(repairs, width="stretch", hide_index=True)
    else:
        st.info("Select a single seed in the sidebar to see the per-event repair log.")

# ---- Table view -------------------------------------------------------------------------------
with tab_table:
    rows = []
    for attr, label, *_ in STRATEGIES:
        key = SUMMARY_KEYS[attr]
        vals = [s["mean_accuracy"][key] for s in summaries]
        finals = [s["final_accuracy"][key] for s in summaries]
        rows.append({
            "Strategy": label,
            "Mean accuracy": f"{sum(vals) / len(vals) * 100:.1f}%",
            "Min-max across seeds": f"{min(vals) * 100:.1f}-{max(vals) * 100:.1f}%",
            "Final accuracy": f"{sum(finals) / len(finals) * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Per-step telemetry")
    st.dataframe(df, width="stretch", hide_index=True)
