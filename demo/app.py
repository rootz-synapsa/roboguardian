from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVENTS = RESULTS / "events"


@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_jsonl(relative_path: str) -> list[dict[str, Any]]:
    path = ROOT / relative_path
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.0f}%"


def metric_ms(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f} ms"


def source_link(path: str) -> str:
    return f"`{path}`"


st.set_page_config(
    page_title="RoboGuardian Evidence Demo",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #071522; color: #eef4f7; }
    [data-testid="stSidebar"] { background: #0b2032; }
    [data-testid="stMetric"] { background: #102b40; border: 1px solid #21445d; border-radius: 12px; padding: 14px; }
    .hero { border-left: 4px solid #11d9ee; padding: 10px 0 10px 18px; margin: 8px 0 22px; }
    .hero h1 { margin: 0; color: #f5f8fa; font-size: 2.25rem; }
    .hero p { margin: 6px 0 0; color: #9eb2c0; font-size: 1.05rem; }
    .boundary { background: #102b40; border: 1px solid #34536a; border-radius: 12px; padding: 14px 16px; color: #c5d3dc; }
    .ok { color: #00e887; font-weight: 700; }
    .warn { color: #f4c95d; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


g6 = load_json("results/g6_reliability.json")
g7 = load_json("results/g7_openvino.json")
h1 = load_json("results/g8_robustness_matrix.json")
h2 = load_json("results/g8_latency_budget.json")
g3 = load_json("results/g3_failure.json")
summary = g6["summary"]

st.markdown(
    """
    <div class="hero">
      <h1>RoboGuardian</h1>
      <p>Governed robotic execution under controlled world change</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This interactive page replays repository evidence from a controlled MuJoCo simulation. "
    "It does not run real hardware, and it does not claim safety certification."
)

with st.sidebar:
    st.markdown("### Evidence Demo")
    st.caption("Read-only replay of checked-in results")
    st.markdown("**Repository**")
    st.markdown("[rootz-synapsa/roboguardian](https://github.com/rootz-synapsa/roboguardian)")
    st.divider()
    st.markdown("**Evidence sources**")
    for item in (
        "results/g3_failure.json",
        "results/g6_reliability.json",
        "results/g7_openvino.json",
        "results/g8_robustness_matrix.json",
        "results/g8_latency_budget.json",
    ):
        st.caption(item)
    st.divider()
    st.caption("Seed 42 · 10 trials per arm · CPU / MuJoCo simulation")

st.markdown("### Killer result")
cols = st.columns(3)
for column, arm, label, color in zip(
    cols,
    ("A", "B", "C"),
    ("Happy path", "Open loop", "RoboGuardian"),
    ("normal", "inverse", "normal"),
):
    data = summary[arm]
    with column:
        st.metric(f"Arm {arm} · {label}", f"{pct(data['task_success_rate'])} success")
        stale = pct(data["stale_action_execution_rate"])
        st.caption(f"Stale action executed: {stale}")

st.markdown("### Head-to-head reliability")
reliability_rows = []
for arm, label in (("A", "Happy path"), ("B", "Open loop"), ("C", "RoboGuardian")):
    data = summary[arm]
    reliability_rows.append(
        {
            "Arm": arm,
            "Controller": label,
            "Trials": data["n_trials"],
            "Task success": pct(data["task_success_rate"]),
            "Disturbance detected": pct(data["disturbance_detection_rate"]),
            "Recovery success": pct(data["recovery_success_rate"]),
            "Stale action executed": pct(data["stale_action_execution_rate"]),
        }
    )
st.table(reliability_rows)
st.caption("Source: results/g6_reliability.json")

st.divider()

left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Replay a recorded trial")
    scenario = st.selectbox(
        "Choose an evidence trace",
        ("Governed recovery", "Open-loop baseline"),
        label_visibility="collapsed",
    )
    trial_number = st.slider("Trial index", min_value=0, max_value=9, value=0)
    replay = st.button("Replay selected evidence", type="primary", width="stretch")

    if replay:
        if scenario == "Governed recovery":
            trace_path = "results/events/G5-RECOVERY-simulation-seed42.jsonl"
            events = load_jsonl(trace_path)
            matching = [row for row in events if row.get("trial_index") == trial_number]
            st.success("RECOVERED · fresh action executed")
            st.write(
                "The recorded trace shows a fresh re-observation and a fresh target. "
                "The stale target remains traceable but is not executed."
            )
        else:
            trace_path = "results/events/G3-BASELINE-seed42.jsonl"
            events = load_jsonl(trace_path)
            matching = [row for row in events if row.get("trial_index") == trial_number]
            st.error("FAILED · stale plan continued without recovery")
            st.write(
                "The recorded baseline trace shows execution under the displaced-object condition "
                "without a recovery path."
            )
        st.caption(f"Trace source: {trace_path}")
        if matching:
            compact = []
            for row in matching[-4:]:
                compact.append(
                    {
                        "step": row.get("step_id"),
                        "state": row.get("state"),
                        "decision": row.get("decision"),
                        "executed": row.get("executed"),
                        "task_failed": row.get("task_failed"),
                        "stale_action_executed": row.get("stale_action_executed"),
                        "fresh_action_executed": row.get("fresh_action_executed"),
                    }
                )
            st.dataframe(compact, width="stretch", hide_index=True)
        else:
            st.warning("No matching event was found for this trial index.")

with right:
    st.markdown("### Controlled failure")
    st.metric("Open-loop failures", f"{g3['n_failed']}/{g3['n_total']}")
    st.caption(f"Reproducibility rate: {pct(g3['reproducibility_rate'])}")
    st.markdown(
        "The plate displacement is **+0.10 m along X** and the baseline ends "
        "**0.30 m off target** under the tested condition."
    )
    with st.expander("Show raw G3 evidence"):
        st.json(g3, expanded=False)

st.divider()

st.markdown("### Hardening evidence")
h1_col, h2_col = st.columns(2)
with h1_col:
    st.markdown("#### H1 · Robustness matrix")
    condition_rows = []
    for name, condition in h1["conditions"].items():
        s = condition["summary"]
        condition_rows.append(
            {
                "Condition": name.replace("_", " "),
                "Trials": s["n_trials"],
                "Task success": pct(s["task_success_rate"]),
                "Safe stop": pct(s["safe_stop_rate"]),
                "Stale executed": pct(s["stale_action_execution_rate"]),
                "Pass": "PASS" if s["condition_pass"] else "FAIL",
            }
        )
    st.dataframe(condition_rows, width="stretch", hide_index=True)
    st.caption("Delay entries are observation-delay proxies, not fixed-rate camera latency measurements.")
    st.caption("Source: results/g8_robustness_matrix.json")

with h2_col:
    st.markdown("#### H2 · Control-loop latency")
    latency = h2["summary_ms"]
    latency_rows = [
        {"Component": "Observe → decision", "p50": metric_ms(latency["observe_to_decision_ms"]["p50_ms"]), "p95": metric_ms(latency["observe_to_decision_ms"]["p95_ms"])},
        {"Component": "IK / re-plan", "p50": metric_ms(latency["replan_ms"]["p50_ms"]), "p95": metric_ms(latency["replan_ms"]["p95_ms"])},
        {"Component": "Recovery total", "p50": metric_ms(latency["recovery_total_ms"]["p50_ms"]), "p95": metric_ms(latency["recovery_total_ms"]["p95_ms"])},
    ]
    st.table(latency_rows)
    st.caption("Measured inside the Python/MuJoCo simulation provider; not an end-to-end camera-to-actuator path.")
    st.caption("Source: results/g8_latency_budget.json")

st.divider()

st.markdown("### G7 · Perception boundary")
g7a, g7b, g7c, g7d = st.columns(4)
g7a.metric("Mean error", "~0.7 cm")
g7b.metric("Worst case", "2.4 cm")
g7c.metric("ONNX p50", metric_ms(g7["baseline_onnxruntime_cpu"]["p50_ms"]))
g7d.metric("OpenVINO p50", metric_ms(g7["openvino"]["p50_ms"]))
st.warning("OpenVINO measured 0.44× the ONNX Runtime p50 result on this tiny model. No speedup claim is made.")

with st.expander("Show raw G7 evidence"):
    st.json(g7, expanded=False)

st.markdown("### What this demo does not claim")
st.markdown(
    """
    <div class="boundary">
    This is a controlled simulation evidence replay. It is not a live physical SO-101 demonstration, """
    """does not establish end-to-end camera-to-actuator latency, and does not provide safety certification. """
    """The next validation steps are stochastic disturbances, end-to-end vision recovery, and physical SO-101 validation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("RoboGuardian · evidence-first demo · MIT License")
