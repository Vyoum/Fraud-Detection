"""Live Kafka decisions, read through SQLite without loading the ML pipeline."""

import sqlite3

import pandas as pd
import streamlit as st

from src.streaming.config import decision_db
from src.streaming.store import read_decisions


@st.fragment(run_every="2s")
def render_live():
    st.caption("Incoming transaction decisions. Scores come from the saved model; actions come from policy.py.")
    try:
        counts, records = read_decisions(decision_db())
    except sqlite3.OperationalError as exc:
        st.warning(f"Decision store is not ready: {exc}")
        return
    columns = st.columns(4)
    columns[0].metric("Decisions received", sum(counts.values()))
    for column, action in zip(columns[1:], ("allow", "step_up", "review")):
        column.metric(action.replace("_", " ").title(), counts.get(action, 0))
    if not records:
        st.info("Waiting for transaction decisions. Start the streaming services and replay described in docs/STREAMING.md.")
        return
    frame = pd.DataFrame(records)
    actions = st.multiselect("Actions", ["allow", "step_up", "review"],
                            default=["allow", "step_up", "review"])
    visible = frame[frame["action"].isin(actions)]
    st.dataframe(visible[["transaction_id", "transaction_dt", "uid", "cluster_id", "score", "action",
                          "threshold_applied", "timestamp"]], hide_index=True, use_container_width=True)
    st.caption("Latest 200 distinct decisions. Replaying the same event with the same bundle is counted once.")
    if len(visible):
        selected = st.selectbox("Inspect decision", visible["decision_id"].tolist(),
                                format_func=lambda value: f"Transaction {frame.set_index('decision_id').loc[value, 'transaction_id']} · {value[:12]}")
        st.json(next(record for record in records if record["decision_id"] == selected))
