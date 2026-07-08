#!/usr/bin/env python3
"""
FleetSense Web — Streamlit version of the predictive maintenance app.


"""

import os

import pandas as pd
import streamlit as st

from predictive_maintenance import Fleet, TYPE_BASELINES
from ml_model import (
    train as train_ml_model,
    predict_failure_probability,
    add_labeled_example,
    count_user_examples,
    get_retrain_history,
)

st.set_page_config(page_title="FleetSense", page_icon="🛠️", layout="wide")

# A real file on disk, not a variable baked into the source. Every machine,
# reading, and reported failure lands here automatically — close the app,
# restart your computer, come back next week, and it's still there.
PERSIST_PATH = os.path.join(os.path.dirname(__file__), "fleet_state.json")

STATUS_COLORS = {
    "healthy": "#34D3AC",
    "watch": "#5EC8E8",
    "warning": "#F2A93B",
    "critical": "#F0524A",
}


def status_for_score(score: float) -> str:
    if score >= 80:
        return "healthy"
    if score >= 60:
        return "watch"
    if score >= 35:
        return "warning"
    return "critical"


def persist():
    """Call this after any change so the fleet survives a restart — this is
    the actual memory, not st.session_state (which Streamlit throws away
    whenever the app process restarts)."""
    st.session_state.fleet.save(PERSIST_PATH)


# ---------------------------------------------------------------------------
# Fleet lives in session_state for fast access during a run, but is loaded
# from — and saved back to — a real file on disk, so it's remembered across
# restarts instead of being rebuilt from the seed function every time.
# ---------------------------------------------------------------------------

if "fleet" not in st.session_state:
    if os.path.exists(PERSIST_PATH):
        st.session_state.fleet = Fleet.load(PERSIST_PATH)
    else:
        fleet = Fleet()
        fleet.seed_demo_history(machines_per_type=6)
        st.session_state.fleet = fleet
        persist()

fleet: Fleet = st.session_state.fleet


@st.cache_resource
def get_trained_model(n_user_examples: int):
    """Cached on n_user_examples — when that number changes (a new real
    outcome was reported), Streamlit's cache misses and this actually
    retrains, rather than silently reusing the old model."""
    return train_ml_model()


def owned_machines():
    return [m for m in fleet.machines.values() if m.owner != "fleet-history"]


# ---------------------------------------------------------------------------
# Sidebar — add machines / log readings / persistence
# ---------------------------------------------------------------------------

st.sidebar.title("🛠️ FleetSense")
st.sidebar.caption("Multi-site predictive maintenance")

with st.sidebar.expander("➕ Add a machine", expanded=False):
    with st.form("add_machine_form", clear_on_submit=True):
        mid = st.text_input("Machine ID (unique)")
        owner = st.text_input("Owner name")
        name = st.text_input("Display name")
        mtype = st.selectbox("Machine type", list(TYPE_BASELINES))
        site = st.text_input("Site")
        submitted = st.form_submit_button("Add machine")
        if submitted:
            if not all([mid, owner, name, site]):
                st.error("All fields are required.")
            elif mid in fleet.machines:
                st.error(f"Machine id '{mid}' already exists.")
            else:
                fleet.add_machine(mid, owner, name, mtype, site)
                persist()
                st.success(f"Added {name}.")
                st.rerun()

with st.sidebar.expander("📈 Log a sensor reading", expanded=False):
    ids = [m.machine_id for m in owned_machines()]
    if not ids:
        st.info("Add a machine first.")
    else:
        with st.form("log_reading_form", clear_on_submit=True):
            target = st.selectbox("Machine", ids)
            hours = st.number_input("Runtime hours since install", min_value=0.0, step=100.0)
            vibration = st.number_input("Vibration (mm/s)", min_value=0.0, step=0.1)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, step=1.0)
            current = st.number_input("Current draw (% of rated)", min_value=0.0, step=1.0)
            submitted = st.form_submit_button("Log reading")
            if submitted:
                fleet.add_reading(target, hours, vibration, temperature, current)
                persist()
                st.success("Reading logged.")
                st.rerun()

with st.sidebar.expander("⚠️ Record an actual failure", expanded=False):
    ids = [m.machine_id for m in owned_machines()]
    if not ids:
        st.info("Add a machine first.")
    else:
        with st.form("record_failure_form", clear_on_submit=True):
            target = st.selectbox("Machine", ids, key="fail_target")
            actual_hours = st.number_input("Runtime hours at actual failure", min_value=0.0, step=100.0)
            submitted = st.form_submit_button("Record failure")
            if submitted:
                fleet.record_actual_failure(target, actual_hours)
                persist()
                st.success("Recorded. This machine now counts toward the accuracy backtest.")
                st.rerun()

st.sidebar.divider()
save_path = "fleet_export.json"
if st.sidebar.button("💾 Export fleet as JSON"):
    fleet.save(save_path)
    with open(save_path, "rb") as f:
        st.sidebar.download_button("Download fleet_export.json", f, file_name="fleet_export.json")

uploaded = st.sidebar.file_uploader("📂 Load a fleet JSON", type="json")
if uploaded is not None:
    with open("uploaded_fleet.json", "wb") as f:
        f.write(uploaded.read())
    st.session_state.fleet = Fleet.load("uploaded_fleet.json")
    persist()
    st.rerun()

st.sidebar.caption(f"💾 Auto-saved to `{os.path.basename(PERSIST_PATH)}` after every change — "
                    "this data is remembered even if you close the app or restart your computer.")
if st.sidebar.button("🗑️ Reset to fresh demo fleet"):
    fresh = Fleet()
    fresh.seed_demo_history(machines_per_type=6)
    st.session_state.fleet = fresh
    persist()
    st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

tab_overview, tab_detail, tab_accuracy, tab_ml = st.tabs(
    ["📊 Fleet Overview", "🔍 Machine Detail", "🎯 Prediction Accuracy", "🤖 Real ML Model"]
)

# ---- Overview ---------------------------------------------------------

with tab_overview:
    machines = owned_machines()
    if not machines:
        st.info("No machines yet. Use **Add a machine** in the sidebar to get started.")
    else:
        rows = []
        for m in machines:
            p = fleet.predict(m.machine_id)
            rows.append({
                "Machine": m.name,
                "ID": m.machine_id,
                "Type": m.machine_type,
                "Site": m.site,
                "Health": round(p.health_score, 1),
                "Status": status_for_score(p.health_score),
                "Horizon (days)": round(p.predicted_days_to_failure) if p.predicted_days_to_failure else None,
                "Confidence": p.confidence,
                "Readings": p.own_reading_count,
            })
        df = pd.DataFrame(rows)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines monitored", len(df))
        c2.metric("Avg fleet health", f"{df['Health'].mean():.0f}")
        at_risk = (df["Horizon (days)"].fillna(9999) < 30).sum()
        c3.metric("Need attention (<30d)", int(at_risk))
        c4.metric("Sites", df["Site"].nunique())

        def highlight(row):
            color = STATUS_COLORS[row["Status"]]
            return [f"background-color: {color}22"] * len(row)

        st.dataframe(
            df.style.apply(highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(df.set_index("Machine")["Health"])

# ---- Detail -------------------------------------------------------------

with tab_detail:
    machines = owned_machines()
    if not machines:
        st.info("No machines yet.")
    else:
        options = {f"{m.name} ({m.machine_id})": m.machine_id for m in machines}
        choice = st.selectbox("Choose a machine", list(options))
        mid = options[choice]
        machine = fleet.machines[mid]
        p = fleet.predict(mid)
        status = status_for_score(p.health_score)

        st.subheader(machine.name)
        st.caption(f"{machine.machine_type} · {machine.site} · owner: {machine.owner}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Health score", f"{p.health_score:.1f}/100")
        c2.metric("Failure window", f"~{p.predicted_days_to_failure:.0f}d" if p.predicted_days_to_failure else "None near-term")
        c3.metric("Confidence", p.confidence, help=f"Cohort: {p.cohort_size} similar machines · Own readings: {p.own_reading_count}")

        st.markdown(
            f"""<div style="height:14px;border-radius:8px;overflow:hidden;display:flex;">
            <div style="flex:22;background:#34D3AC44"></div>
            <div style="flex:30;background:#5EC8E844"></div>
            <div style="flex:28;background:#F2A93B44"></div>
            <div style="flex:20;background:#F0524A44"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#888;">
            <span>Stable</span><span>Monitor</span><span>Elevated</span><span>Imminent</span>
            </div>""",
            unsafe_allow_html=True,
        )

        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Why it was flagged**")
            if p.risk_factors:
                for r in p.risk_factors:
                    st.write(f"- {r}")
            else:
                st.write("No readings currently deviate from fleet norms.")
        with colR:
            st.markdown("**Recommended to prevent failure**")
            if p.recommended_actions:
                for a in p.recommended_actions:
                    st.write(f"- {a}")
            else:
                st.write("No action needed right now.")

        history = fleet.health_history(mid)
        if history:
            hist_df = pd.DataFrame(history).set_index("hours")
            st.markdown("**Health score over this machine's own history**")
            st.line_chart(hist_df)
        else:
            st.caption("Log a reading to start building this machine's own trend line.")

# ---- Accuracy backtest --------------------------------------------------

with tab_accuracy:
    st.markdown(
        "This replays every machine that has actually failed (seeded fleet history, "
        "plus any you've recorded with **Record an actual failure**) and checks how far "
        "off the prediction was, using only what was known at each point in time. "
        "If the cohort-learning idea works, error should trend down as more readings accumulate."
    )
    type_filter = st.selectbox("Machine type", ["All types"] + list(TYPE_BASELINES))
    mtype = None if type_filter == "All types" else type_filter
    rows = fleet.backtest_accuracy(mtype)
    if not rows:
        st.info("No failed machines to backtest yet for this type.")
    else:
        bdf = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.metric("Backtested predictions", len(bdf))
        c2.metric("Avg error", f"{bdf['pct_error'].mean():.1f}%")

        by_readings = bdf.groupby("readings_used")["pct_error"].mean()
        st.markdown("**Average % error vs. number of readings used**")
        st.line_chart(by_readings)
        st.caption("Fewer readings (left) rely mostly on the cohort; more readings (right) let the model use this machine's own trend.")

        with st.expander("Raw backtest rows"):
            st.dataframe(bdf, use_container_width=True, hide_index=True)

# ---- Real ML model -------------------------------------------------------

with tab_ml:
    st.markdown(
        "Everything above (Overview / Detail / Accuracy) runs on a **hand-written formula** — "
        "an exponential decay curve compared against simulated cohort statistics. It's a reasonable "
        "engineering model, but it isn't machine learning: nothing in it was *learned* from data.\n\n"
        "This tab is different. It's a `RandomForestClassifier` actually trained on the "
        "**[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** "
        "(Matzka, 2020) — a published, peer-reviewed benchmark used across real predictive-maintenance "
        "research, logged from a real milling-machine rig with genuine sensor ranges and a realistic "
        "3.4% failure rate. The model has never seen the test rows below during training.\n\n"
        "**This one actually retrains.** Every time you report a real outcome below, it gets added "
        "to the training set and the model is refit from scratch on base data + everything reported "
        "so far — the model running after that is a genuinely different, re-fit model, not the same "
        "one relabeled."
    )

    n_user = count_user_examples()
    tm = get_trained_model(n_user)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trained on", f"{tm.n_train:,} rows")
    c2.metric("Tested on (held out)", f"{tm.n_test:,} rows")
    c3.metric("ROC-AUC on test set", f"{tm.roc_auc:.3f}")
    c4.metric("Real outcomes you've reported", n_user)

    st.markdown("**What the model actually learned mattered most** (not asserted by a human):")
    imp_df = pd.DataFrame(tm.feature_importances, columns=["Feature", "Importance"]).set_index("Feature")
    st.bar_chart(imp_df)

    with st.expander("Full held-out classification report"):
        st.code(tm.test_report)
        st.caption(f"Confusion matrix [[TN, FP], [FN, TP]]: {tm.confusion} — "
                    "with only 3.4% of machines actually failing, catching true failures without "
                    "drowning in false alarms is the hard part, which is why recall/precision matter "
                    "more here than raw accuracy.")

    st.divider()
    st.markdown("**Try it — enter live operating parameters and get the trained model's real-time prediction:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp = st.number_input("Air temperature (K)", value=300.0, step=0.5, key="pred_air")
        process_temp = st.number_input("Process temperature (K)", value=310.0, step=0.5, key="pred_proc")
    with col2:
        rpm = st.number_input("Rotational speed (rpm)", value=1450.0, step=10.0, key="pred_rpm")
        torque = st.number_input("Torque (Nm)", value=45.0, step=1.0, key="pred_torque")
    with col3:
        tool_wear = st.number_input("Tool wear (min)", value=100.0, step=5.0, key="pred_wear")
        ptype = st.selectbox("Product quality variant", ["L", "M", "H"], key="pred_type")

    if st.button("Run real ML prediction"):
        prob = predict_failure_probability(tm, air_temp, process_temp, rpm, torque, tool_wear, ptype)
        st.metric("Predicted failure probability", f"{prob:.1%}")
        if prob > 0.5:
            st.error("High risk — these parameters resemble past real failures in the training data.")
        elif prob > 0.15:
            st.warning("Elevated risk — worth monitoring.")
        else:
            st.success("Low risk based on the trained model.")

    st.divider()
    st.markdown("**📝 Report a real outcome — this is what actually retrains the model:**")
    st.caption(
        "Enter what a machine's readings actually looked like, and whether it actually failed. "
        "This gets appended to the training data permanently and folded into the next retrain — "
        "including if you were reporting the SAME machine you just predicted on above, to see "
        "whether the model was right."
    )
    with st.form("report_outcome_form", clear_on_submit=True):
        r1, r2, r3 = st.columns(3)
        with r1:
            o_air = st.number_input("Air temperature (K)", value=300.0, step=0.5, key="outcome_air")
            o_proc = st.number_input("Process temperature (K)", value=310.0, step=0.5, key="outcome_proc")
        with r2:
            o_rpm = st.number_input("Rotational speed (rpm)", value=1450.0, step=10.0, key="outcome_rpm")
            o_torque = st.number_input("Torque (Nm)", value=45.0, step=1.0, key="outcome_torque")
        with r3:
            o_wear = st.number_input("Tool wear (min)", value=100.0, step=5.0, key="outcome_wear")
            o_type = st.selectbox("Product quality variant", ["L", "M", "H"], key="outcome_type")
        o_failed = st.radio("What actually happened?", ["Did NOT fail", "DID fail"], horizontal=True)
        submitted = st.form_submit_button("Add outcome and retrain")
        if submitted:
            add_labeled_example(o_air, o_proc, o_rpm, o_torque, o_wear, o_type, o_failed == "DID fail")
            st.success("Outcome recorded. Retraining on the updated dataset...")
            st.rerun()

    history = get_retrain_history()
    if len(history) > 1:
        st.markdown("**Retrain history — does accuracy actually move as real data comes in?**")
        hdf = pd.DataFrame(history).set_index("version")[["roc_auc", "accuracy"]]
        st.line_chart(hdf)
        st.caption(
            "Each point is a real retrain, not a simulated one. With only a handful of real examples "
            "added on top of a solid 10,000-row base, don't expect dramatic swings — and it's not "
            "guaranteed to always go up, since a single new example can pull the model in an odd "
            "direction before more data smooths it out. That's genuine model behavior, not a display bug."
        )
        with st.expander("Raw retrain history"):
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    elif len(history) == 1:
        st.info("Only one training run logged so far — report an outcome above to see the history chart appear.")

    st.caption(
        "Note on scope: this dataset's features (torque, RPM, tool wear) are specific to milling "
        "machines, not a universal fit for every machine type in the fleet above. To use a real "
        "trained model for pumps, compressors, etc., you'd want a dataset logged from that "
        "equipment type — the training approach here would carry over directly."
    )
