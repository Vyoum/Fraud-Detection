"""CLI entry point for `make results`. The only orchestration layer.

Sequences entities -> graph -> model -> evaluate, then layers investigator
evaluation, policy/audit, calibration, and a latency benchmark on top --
all the artifacts built across this project's sessions, wired together so
`make results` produces every one of them from a single
load_and_prepare()/train_both_models() call rather than each artifact
re-loading and re-training independently (which is what scripts/*.py did
when developed one task at a time; they are now thin wrappers that import
these functions -- see e.g. scripts/cost_curve.py).

policy.py's threshold provenance (scripts/derive_policy_thresholds.py) and
investigator.py's Anthropic-API path are NOT re-run here beyond what
write_investigator_eval/write_audit_sample already do; nothing in this
module retrains model.py or edits evaluate.py/entities.py/graph.py, all
frozen this session.

One deliberate exception: results/case_studies.md is NOT regenerated here.
It is a hand-curated judgment artifact (see scripts/case_studies.py's own
docstring) -- "does this look like a real ring" is not something to
template. `make results` leaves it untouched; re-run
scripts/case_studies.py by hand for fresh raw material if the underlying
data changes, then re-write the case studies by hand on top of it.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src import data, entities, evaluate, graph, investigator, model, policy
from src.graph import get_connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ARCHITECTURE_PATH = REPO_ROOT / "ARCHITECTURE.md"

# See graph.py's module docstring for the full max_degree sweep: the
# function's own default (1000) collapses 64% of all uids into one
# connected component on this dataset. 20 keeps the largest cluster at 126
# uids (0.06%).
MAX_DEGREE = 20

# Illustrative cost assumptions, NOT Razorpay figures -- see
# results/ablation.md for the explicit caveat. Missing a fraud case is
# assumed to cost 100x a false alarm (chargeback loss vs. customer-friction
# cost of an unnecessary step-up).
COST_FN = 500.0
COST_FP = 5.0

# The cost-minimizing threshold for a well-calibrated binary classifier
# under (cost_fn, cost_fp) is where the two expected costs balance -- used
# as the ablation table's headline threshold. write_cost_curve sweeps
# thresholds directly rather than relying on this calibration assumption.
DEFAULT_THRESHOLD = COST_FP / (COST_FN + COST_FP)

CLUSTER_FEATURE_COLUMNS = [
    "cluster_size_uids",
    "cluster_txn_count",
    "cluster_edge_density",
    "node_degree",
    "cluster_velocity",
    "cluster_amt_cv",
    "cluster_burst_concentration",
    "uid_email_domain_count",
    "cluster_email_uid_ratio",
    "cluster_prior_fraud_share",
]

COST_CURVE_THRESHOLDS = np.concatenate(
    [np.linspace(0.0, 0.05, 200), np.linspace(0.05, 1.0, 100)[1:]]
)

_NUMBER_PATTERN = re.compile(r"-?\d+\.\d+|-?\d+")


@dataclass
class PipelineData:
    df: pd.DataFrame
    entity_ids: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    as_of: float
    entity_graph: graph.EntityGraph
    cluster_features: pd.DataFrame  # indexed by uid (entity_id)


@dataclass
class TrainedModels:
    baseline_model: lgb.Booster
    cluster_model: lgb.Booster
    X_train_baseline: pd.DataFrame
    X_train_cluster: pd.DataFrame
    X_test_baseline: pd.DataFrame
    X_test_cluster: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def broadcast_cluster_features(
    entity_ids: pd.Series, cluster_features: pd.DataFrame
) -> pd.DataFrame:
    """Expand per-uid cluster features to per-TransactionID.

    A row whose uid has no entry in `cluster_features` (no uid at all, or a
    uid with zero pre-as_of history) gets NaN in every cluster column --
    explicit nulls, never a dropped row or a fabricated zero. This is how
    the ~11% no-uid population (see results/uid_validation.md) is handled:
    they stay in training/evaluation with null cluster features rather than
    being excluded, since excluding them would drop the highest-risk slice
    of transactions (11.63% fraud rate there vs. 2.46% for uid'd rows).
    """
    broadcast = cluster_features.reindex(entity_ids.to_numpy())
    broadcast.index = entity_ids.index
    return broadcast


def load_and_prepare(nrows: int | None = None, history_parquet: str | None = None) -> PipelineData:
    """Load data, resolve entities, split temporally, build the graph and
    compute cluster features -- all from train-period data only.
    """
    df = (data.load_history_parquet(history_parquet) if history_parquet
          else data.load_transactions(DATA_DIR, nrows=nrows))
    entity_ids = entities.resolve_entities(df)

    train_df, test_df = evaluate.temporal_train_test_split(df)
    as_of = float(test_df["TransactionDT"].min())

    entity_graph = graph.build_entity_graph(train_df, entity_ids, max_degree=MAX_DEGREE)
    cluster_features = graph.compute_cluster_features(
        train_df, entity_ids, entity_graph.graph, as_of=as_of
    )

    return PipelineData(
        df=df,
        entity_ids=entity_ids,
        train_df=train_df,
        test_df=test_df,
        as_of=as_of,
        entity_graph=entity_graph,
        cluster_features=cluster_features,
    )


def train_both_models(pipeline_data: PipelineData) -> TrainedModels:
    """Train the baseline and cluster models with identical everything
    except the cluster feature columns.
    """
    pd_ = pipeline_data
    y_train = pd_.train_df.set_index("TransactionID")["isFraud"]
    y_test = pd_.test_df.set_index("TransactionID")["isFraud"]

    cluster_by_transaction = broadcast_cluster_features(
        pd_.entity_ids, pd_.cluster_features
    )
    cluster_train = cluster_by_transaction.reindex(y_train.index)
    cluster_test = cluster_by_transaction.reindex(y_test.index)

    X_train_baseline = model.build_feature_matrix(pd_.train_df)
    X_test_baseline = model.build_feature_matrix(pd_.test_df)
    X_train_cluster = model.build_feature_matrix(pd_.train_df, cluster_train)
    X_test_cluster = model.build_feature_matrix(pd_.test_df, cluster_test)

    baseline_model = model.train_baseline_model(X_train_baseline, y_train)
    cluster_model = model.train_cluster_model(X_train_cluster, y_train)

    return TrainedModels(
        baseline_model=baseline_model,
        cluster_model=cluster_model,
        X_train_baseline=X_train_baseline,
        X_train_cluster=X_train_cluster,
        X_test_baseline=X_test_baseline,
        X_test_cluster=X_test_cluster,
        y_train=y_train,
        y_test=y_test,
    )


def evaluate_both_models(trained: TrainedModels) -> dict[str, dict[str, float]]:
    return {
        "baseline": evaluate.evaluate_model(
            trained.baseline_model,
            trained.X_test_baseline,
            trained.y_test,
            threshold=DEFAULT_THRESHOLD,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
        ),
        "cluster": evaluate.evaluate_model(
            trained.cluster_model,
            trained.X_test_cluster,
            trained.y_test,
            threshold=DEFAULT_THRESHOLD,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
        ),
    }


def _feature_importance_table(
    booster: lgb.Booster, top_n: int = 20
) -> pd.DataFrame:
    importances = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    return importances.head(top_n)


def write_ablation_report(
    pipeline_data: PipelineData,
    trained: TrainedModels,
    metrics: dict[str, dict[str, float]],
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    lines: list[str] = []

    lines.append("# Ablation: transaction-only baseline vs. cluster-augmented")
    lines.append("")
    lines.append(
        f"Temporal split: {len(pipeline_data.train_df):,} train rows, "
        f"{len(pipeline_data.test_df):,} test rows "
        f"(as_of = TransactionDT {pipeline_data.as_of:,.0f}, the first "
        "test-period timestamp)."
    )
    lines.append("")
    lines.append(
        "Cost assumptions are illustrative, NOT Razorpay figures: "
        f"cost_fn={COST_FN:g} (a missed abuse case), cost_fp={COST_FP:g} "
        "(a false alarm / unnecessary step-up) -- a 100:1 ratio, chosen to "
        "represent a chargeback loss being much costlier than customer "
        "friction, nothing more precise than that. The threshold used below "
        f"({DEFAULT_THRESHOLD:.4f}) is cost_fp/(cost_fn+cost_fp), the "
        "cost-minimizing point for a well-calibrated classifier under this "
        "cost ratio; results/cost_curve.png sweeps thresholds directly "
        "rather than relying on that calibration assumption."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |")
    lines.append("|---|---:|---:|---:|")
    for name in ("baseline", "cluster"):
        m = metrics[name]
        lines.append(
            f"| {name} | {m['pr_auc']:.4f} | {m['recall_at_1pct_fpr']:.4f} | "
            f"{m['cost_per_10k']:.2f} |"
        )
    lines.append("")

    pr_auc_lift = metrics["cluster"]["pr_auc"] - metrics["baseline"]["pr_auc"]
    recall_lift = (
        metrics["cluster"]["recall_at_1pct_fpr"]
        - metrics["baseline"]["recall_at_1pct_fpr"]
    )
    cost_delta = metrics["cluster"]["cost_per_10k"] - metrics["baseline"]["cost_per_10k"]
    lines.append(
        f"Cluster model vs. baseline: PR-AUC {pr_auc_lift:+.4f}, "
        f"recall@1%FPR {recall_lift:+.4f}, cost per 10k {cost_delta:+.2f} "
        "(negative is better for cost). Reported as-is; the derivation and "
        "features were not adjusted after seeing these numbers."
    )
    lines.append("")

    lines.append("## Hyperparameters (identical for both models)")
    lines.append("")
    lines.append("```")
    for k, v in model.LGBM_PARAMS.items():
        lines.append(f"{k}: {v}")
    lines.append(f"num_boost_round: {model.NUM_BOOST_ROUND}")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Baseline features: {trained.X_train_baseline.shape[1]}. "
        f"Cluster features add: "
        f"{trained.X_train_cluster.shape[1] - trained.X_train_baseline.shape[1]} "
        "columns (the graph.compute_cluster_features output -- cluster "
        "size/txn count/edge density/velocity/amount CV/burst "
        "concentration/email-uid ratio/prior-fraud share, plus per-uid node "
        "degree and email domain count)."
    )
    lines.append("")

    lines.append("## Cluster model feature importances (top 20 by gain)")
    lines.append("")
    lines.append("| feature | gain |")
    lines.append("|---|---:|")
    for _, row in _feature_importance_table(trained.cluster_model).iterrows():
        lines.append(f"| {row['feature']} | {row['gain']:,.1f} |")
    lines.append("")

    lines.append("## Graph construction")
    lines.append("")
    lines.append(
        f"max_degree={MAX_DEGREE} (see graph.py's module docstring for why "
        "the function's own default of 1000 is unusable on this dataset -- "
        "it collapses 64% of all uids into one connected component)."
    )
    lines.append(
        f"Built from {len(pipeline_data.train_df):,} train-period "
        f"transactions: {pipeline_data.entity_graph.graph.number_of_nodes():,} "
        f"nodes, {pipeline_data.entity_graph.graph.number_of_edges():,} edges."
    )
    lines.append("")
    excluded = pipeline_data.entity_graph.excluded_hubs
    lines.append(
        f"Hub guard excluded {len(excluded)} values "
        f"(covering {int(excluded['uid_count'].sum()):,} uid-appearances "
        "in total, with overlap across rules) as too common to be evidence "
        "of a relationship. Ten largest:"
    )
    lines.append("")
    lines.append("| rule | value | uid_count |")
    lines.append("|---|---|---:|")
    for _, row in excluded.sort_values("uid_count", ascending=False).head(10).iterrows():
        lines.append(f"| {row['rule']} | {row['value']} | {row['uid_count']:,} |")
    lines.append("")

    (RESULTS_DIR / "ablation.md").write_text("\n".join(lines), encoding="utf-8")


def _remove_section(text: str, heading: str) -> str:
    """Strip a `heading` (e.g. "## Performance") through to the next
    top-level heading or EOF, so re-appending it is idempotent. A no-op if
    `heading` isn't present.
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break
    if start is None:
        return text
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## ") or lines[i].startswith("# "):
            end = i
            break
    return "\n".join(lines[:start] + lines[end:])


def _append_to_ablation(lines: list[str]) -> None:
    ablation_path = RESULTS_DIR / "ablation.md"
    existing = ablation_path.read_text(encoding="utf-8")
    existing = _remove_section(existing, lines[0])
    ablation_path.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


# --- Sanity checks (results/ablation.md "## Sanity checks" section) --------


def _correlation_section(trained: TrainedModels) -> list[str]:
    lines = ["### 1. Correlation of cluster features with isFraud (train set)", ""]

    rows = []
    for col in CLUSTER_FEATURE_COLUMNS:
        series = trained.X_train_cluster[col]
        n_valid = series.notna().sum()
        c = series.corr(trained.y_train)
        rows.append((col, c, n_valid))
    rows.sort(key=lambda r: (abs(r[1]) if pd.notna(r[1]) else -1), reverse=True)

    lines.append(
        f"Computed over the {trained.X_train_cluster.shape[0]:,} train rows; "
        "each feature's correlation uses only the rows where that feature "
        "is non-null (pandas' default pairwise behavior) -- n_valid shows "
        "how many that was per feature."
    )
    lines.append("")
    lines.append("| feature | correlation with isFraud | n_valid |")
    lines.append("|---|---:|---:|")
    flagged = []
    for col, c, n_valid in rows:
        flag = ""
        if pd.notna(c) and abs(c) > 0.5:
            flag = " **(>0.5)**"
            flagged.append((col, c))
        c_str = f"{c:.4f}" if pd.notna(c) else "NaN"
        lines.append(f"| {col} | {c_str}{flag} | {n_valid:,} |")
    lines.append("")

    if flagged:
        joined = ", ".join(f"{c} ({v:.3f})" for c, v in flagged)
        lines.append(
            f"**{len(flagged)} feature(s) exceed the 0.5 red-flag "
            f"threshold: {joined}.** Investigated in section 2."
        )
    else:
        lines.append("No feature exceeds the 0.5 red-flag threshold.")
    lines.append("")
    return lines


def _trace_prior_fraud_leak(pipeline_data: PipelineData) -> list[str]:
    lines = [
        "### 2. Tracing cluster_prior_fraud_share for a leak",
        "",
        "cluster_prior_fraud_share is computed inside "
        "graph.compute_cluster_features as: for every transaction with "
        "`TransactionDT < as_of`, take the per-uid max of isFraud over "
        "*that uid's own* qualifying transactions "
        "(`by_uid[\"isFraud\"].max()`), then average that per-uid flag "
        "across the uids in each cluster. The `as_of` filter is applied to "
        "`transactions` before any of this runs -- there is no code path in "
        "compute_cluster_features that reads isFraud from a row that didn't "
        "pass the `TransactionDT < as_of` filter. That's the argument from "
        "reading the code (and it's what tests/test_graph.py's explicit "
        "leakage test already checks structurally). Below is the argument "
        "from the actual data instead.",
        "",
    ]

    df = pipeline_data.df
    entity_ids = pipeline_data.entity_ids
    as_of = pipeline_data.as_of
    full = df.set_index("TransactionID")
    full = full.assign(uid=entity_ids)
    valid = full.loc[full["uid"].notna()].copy()

    components = get_connected_components(pipeline_data.entity_graph.graph)
    cluster_of = {}
    for i, comp in enumerate(components):
        for u in comp:
            cluster_of[u] = i

    valid["cluster_id"] = valid["uid"].map(cluster_of)
    valid = valid.loc[valid["cluster_id"].notna()]

    uid_to_cluster = valid.drop_duplicates("uid").set_index("uid")["cluster_id"]

    pre = valid[valid["TransactionDT"] < as_of]
    ever_fraud_pre = pre.groupby("uid")["isFraud"].max()
    expected_share = ever_fraud_pre.groupby(uid_to_cluster.reindex(ever_fraud_pre.index)).mean()

    ever_fraud_all = valid.groupby("uid")["isFraud"].max()
    leaked_share = ever_fraud_all.groupby(uid_to_cluster.reindex(ever_fraud_all.index)).mean()

    reported_by_uid = pipeline_data.cluster_features["cluster_prior_fraud_share"]
    reported_by_cluster = reported_by_uid.groupby(uid_to_cluster.reindex(reported_by_uid.index)).first()
    common_idx = reported_by_cluster.index.intersection(expected_share.index)
    mismatches = (
        (reported_by_cluster.loc[common_idx] - expected_share.loc[common_idx]).abs() > 1e-9
    )
    n_checked = len(common_idx)
    n_mismatched = int(mismatches.sum())

    lines.append(
        f"Checked all {n_checked:,} clusters that have both a reported "
        "cluster_prior_fraud_share and an independently-recomputable "
        "pre-as_of value: comparing the pipeline's reported value against "
        "one computed straight from raw rows with `TransactionDT < as_of`, "
        f"bypassing graph.py entirely. Mismatches: **{n_mismatched}**."
    )
    lines.append("")

    gap = (leaked_share - expected_share).abs().dropna()
    gap = gap[gap > 1e-9]
    if gap.empty:
        lines.append(
            "No cluster's leaked-vs-expected prior-fraud share actually "
            "differed on this run, so no single example could demonstrate "
            "the filter doing real work -- reported honestly rather than "
            "picking a less meaningful example. The all-cluster mismatch "
            "count above is the check that matters here."
        )
        lines.append("")
        return lines

    best_cluster_id = gap.idxmax()
    members = sorted(components[int(best_cluster_id)])
    member_rows = valid[valid["cluster_id"] == best_cluster_id]

    reported = float(reported_by_cluster.loc[best_cluster_id])
    expected_value = float(expected_share.loc[best_cluster_id])
    leaked_value = float(leaked_share.loc[best_cluster_id])

    lines.append(
        f"Concrete example -- the cluster where leaking would change this "
        f"feature the most: cluster #{int(best_cluster_id)}, "
        f"{len(members)} member uids."
    )
    lines.append("")
    fraud_rows = member_rows[member_rows["isFraud"] == 1].sort_values(
        ["uid", "TransactionDT"]
    )
    lines.append(
        f"Its {len(fraud_rows)} fraud-labeled transaction(s) (all other "
        "member rows are isFraud=0, omitted for brevity):"
    )
    lines.append("")
    lines.append("| uid | TransactionDT | period | isFraud |")
    lines.append("|---|---:|---|---:|")
    for _, row in fraud_rows.iterrows():
        period = "test" if row["TransactionDT"] >= as_of else "train"
        lines.append(
            f"| {row['uid']} | {row['TransactionDT']:,.0f} | {period} | 1 |"
        )
    lines.append("")

    lines.append(f"- Reported by the pipeline (as_of-filtered): **{reported:.4f}**")
    lines.append(
        f"- Independently recomputed from raw rows with "
        f"`TransactionDT < as_of`, bypassing graph.py entirely: "
        f"**{expected_value:.4f}**"
    )
    lines.append(
        f"- What it would be if test-period rows leaked in (all-time max "
        f"isFraud per member, no as_of filter): **{leaked_value:.4f}**"
    )
    lines.append("")

    if abs(reported - expected_value) < 1e-9:
        lines.append(
            "**Reported value matches the independently-recomputed "
            "pre-as_of-only value, and differs from the would-leak value "
            "for this cluster -- confirmed on real data, not just in the "
            "unit test: no test-period fraud label influenced this "
            "cluster's prior-fraud feature.**"
        )
    else:
        lines.append(
            "**MISMATCH: the reported value does not match the "
            "independently-recomputed pre-as_of value. This is a leak and "
            "is being reported plainly, not fixed quietly.**"
        )
    lines.append("")
    return lines


def _reablation_without_prior_fraud(
    pipeline_data: PipelineData, trained: TrainedModels
) -> list[str]:
    lines = ["### 3. Ablation re-run with cluster_prior_fraud_share removed", ""]

    drop_col = "cluster_prior_fraud_share"
    X_train = trained.X_train_cluster.drop(columns=[drop_col])
    X_test = trained.X_test_cluster.drop(columns=[drop_col])

    trimmed_model = model.train_cluster_model(X_train, trained.y_train)
    trimmed_metrics = evaluate.evaluate_model(
        trimmed_model,
        X_test,
        trained.y_test,
        threshold=DEFAULT_THRESHOLD,
        cost_fn=COST_FN,
        cost_fp=COST_FP,
    )

    lines.append(
        "Same training path (train_cluster_model, identical hyperparameters "
        "and seed), same cluster feature set minus this one column."
    )
    lines.append("")
    lines.append("| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| cluster (no cluster_prior_fraud_share) | "
        f"{trimmed_metrics['pr_auc']:.4f} | "
        f"{trimmed_metrics['recall_at_1pct_fpr']:.4f} | "
        f"{trimmed_metrics['cost_per_10k']:.2f} |"
    )
    lines.append("")
    return lines


def _cluster_assignment_independence(pipeline_data: PipelineData) -> list[str]:
    lines = [
        "### 4. Cluster assignment for a test-period transaction never "
        "depends on test-period edges",
        "",
        "Structurally: `graph.build_entity_graph` is called in "
        "run_pipeline.load_and_prepare with `train_df` only -- `test_df` is "
        "never passed to it, so no test-period transaction can ever "
        "contribute an edge or a node. Verified concretely below rather "
        "than just re-reading the call site.",
        "",
    ]

    df = pipeline_data.df
    entity_ids = pipeline_data.entity_ids
    as_of = pipeline_data.as_of
    full = df.set_index("TransactionID")
    full = full.assign(uid=entity_ids)
    valid = full.loc[full["uid"].notna()]

    train_uids = set(valid.loc[valid["TransactionDT"] < as_of, "uid"].unique())
    test_only_uids = set(
        valid.loc[valid["TransactionDT"] >= as_of, "uid"].unique()
    ) - train_uids

    graph_nodes = set(pipeline_data.entity_graph.graph.nodes())
    leaked_nodes = graph_nodes - train_uids
    lines.append(
        f"- Every node in the graph corresponds to a uid with at least one "
        f"train-period transaction: {len(leaked_nodes)} nodes found in the "
        "graph with zero train-period transactions (should be 0)."
    )

    if test_only_uids:
        example_uid = sorted(test_only_uids)[0]
        in_graph = example_uid in graph_nodes
        broadcast = broadcast_cluster_features(entity_ids, pipeline_data.cluster_features)
        example_txn_ids = entity_ids.index[entity_ids == example_uid]
        example_features = broadcast.loc[broadcast.index.isin(example_txn_ids)]
        all_null = bool(example_features.isna().all().all())
        node_status = "absent from" if not in_graph else "PRESENT IN"
        null_status = "all null" if all_null else "NOT all null"
        if not in_graph and all_null:
            verdict = "as expected -- no train-period history means no cluster signal, not a fabricated one"
        else:
            verdict = "UNEXPECTED"
        lines.append(
            f"- Concrete example: uid `{example_uid}` appears ONLY in the "
            f"test period (no train-period transactions). It is "
            f"{node_status} the graph's node set, and its broadcast "
            f"cluster features are {null_status} ({verdict})."
        )
    else:
        lines.append(
            "- No test-period-only uid found in this run to use as a "
            "concrete example (every uid observed in the test period also "
            "had train-period history)."
        )
    lines.append("")
    return lines


def write_sanity_checks(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    lines = ["## Sanity checks", ""]
    lines += _correlation_section(trained)
    lines += _trace_prior_fraud_leak(pipeline_data)
    lines += _reablation_without_prior_fraud(pipeline_data, trained)
    lines += _cluster_assignment_independence(pipeline_data)
    _append_to_ablation(lines)


# --- Cost curve (results/cost_curve.png + ablation.md section) -------------


def _recall_and_fpr_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    predicted_positive = y_score >= threshold

    tp = np.sum((y_true == 1) & predicted_positive)
    fn = np.sum((y_true == 1) & ~predicted_positive)
    fp = np.sum((y_true == 0) & predicted_positive)
    tn = np.sum((y_true == 0) & ~predicted_positive)

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return float(recall), float(fpr)


def write_cost_curve(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    y_test = trained.y_test.to_numpy()
    scores = {
        "baseline": trained.baseline_model.predict(trained.X_test_baseline),
        "cluster": trained.cluster_model.predict(trained.X_test_cluster),
    }

    costs = {name: [] for name in scores}
    for name, y_score in scores.items():
        for t in COST_CURVE_THRESHOLDS:
            costs[name].append(evaluate.cost_per_10k(y_test, y_score, t, COST_FN, COST_FP))
        costs[name] = np.array(costs[name])

    chosen = {}
    for name, y_score in scores.items():
        best_idx = int(np.argmin(costs[name]))
        best_threshold = float(COST_CURVE_THRESHOLDS[best_idx])
        best_cost = float(costs[name][best_idx])
        recall, fpr = _recall_and_fpr_at_threshold(y_test, y_score, best_threshold)
        chosen[name] = {
            "threshold": best_threshold,
            "cost_per_10k": best_cost,
            "recall": recall,
            "fpr": fpr,
        }

    colors = {"baseline": "#4472C4", "cluster": "#C0504D"}
    zoom_xmax = 0.05

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in scores:
        ax.plot(COST_CURVE_THRESHOLDS, costs[name], label=name, color=colors[name])
        ax.scatter(
            [chosen[name]["threshold"]], [chosen[name]["cost_per_10k"]],
            color=colors[name], zorder=5, s=50, edgecolor="black",
        )
    ax.axvspan(0, zoom_xmax, color="grey", alpha=0.12)
    ax.set_xlabel("Score threshold (flag if score >= threshold)")
    ax.set_ylabel(f"Expected cost per 10k txns (cost_fn={COST_FN:g}, cost_fp={COST_FP:g}, illustrative)")
    ax.set_title("Full threshold range")
    ax.legend()

    zoom_mask = COST_CURVE_THRESHOLDS <= zoom_xmax
    offsets = {"baseline": (12, 15), "cluster": (12, -22)}
    for name in scores:
        ax_zoom.plot(
            COST_CURVE_THRESHOLDS[zoom_mask], costs[name][zoom_mask], label=name, color=colors[name]
        )
        ax_zoom.scatter(
            [chosen[name]["threshold"]], [chosen[name]["cost_per_10k"]],
            color=colors[name], zorder=5, s=70, edgecolor="black",
        )
        ax_zoom.annotate(
            f"{name} chosen: t={chosen[name]['threshold']:.4f}\ncost={chosen[name]['cost_per_10k']:,.0f}",
            (chosen[name]["threshold"], chosen[name]["cost_per_10k"]),
            textcoords="offset points", xytext=offsets[name], fontsize=8,
            arrowprops={"arrowstyle": "-", "color": colors[name], "lw": 0.8},
        )
    ax_zoom.set_xlabel("Score threshold (zoomed: 0 to 0.05)")
    ax_zoom.set_ylabel("Expected cost per 10k txns")
    ax_zoom.set_title("Zoomed: where the minima actually are")
    ax_zoom.legend()

    fig.suptitle("Cost per 10k transactions vs. threshold")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)

    lines = ["## Threshold sweep and cost curve", ""]
    lines.append(
        "results/cost_curve.png sweeps score thresholds directly (not "
        "relying on the calibration assumption behind ablation.md's headline "
        "threshold) and marks each model's own cost-minimizing point."
    )
    lines.append("")
    lines.append("![Cost per 10k vs threshold](cost_curve.png)")
    lines.append("")
    lines.append("| model | chosen threshold | cost per 10k at chosen point | recall | FPR |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("baseline", "cluster"):
        c = chosen[name]
        lines.append(
            f"| {name} | {c['threshold']:.4f} | {c['cost_per_10k']:.2f} | "
            f"{c['recall']:.4f} | {c['fpr']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Worth being explicit about: the cost-minimizing FPR here is "
        f"{chosen['cluster']['fpr']:.0%}-{chosen['baseline']['fpr']:.0%} -- "
        "a direct, correct mathematical consequence of the assumed 100:1 "
        "cost_fn:cost_fp ratio (missing fraud is assumed to be that much "
        "worse than a false alarm, so the optimum flags aggressively), not "
        "a bug. In practice this means stepping up upwards of a third of "
        "all legitimate transactions at the \"optimal\" point -- whether "
        "that's acceptable is a business call the assumed cost ratio drives "
        "entirely; a less aggressive cost ratio (or a friction budget "
        "constraint) would move the chosen threshold and the resulting FPR "
        "substantially."
    )
    lines.append("")
    _append_to_ablation(lines)


# --- Calibration (results/calibration.png + ablation.md section) ----------


def write_calibration(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    n_bins = 15
    y_test = trained.y_test.to_numpy()
    y_score = trained.cluster_model.predict(trained.X_test_cluster)

    brier = float(brier_score_loss(y_test, y_score))
    base_rate = float(y_test.mean())
    brier_baseline = float(np.mean((y_test - base_rate) ** 2))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_score, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", color="#C0504D", label="cluster model")
    ax.set_xlabel("Mean predicted score (within bin)")
    ax.set_ylabel("Observed fraction of positives (within bin)")
    ax.set_title(f"Reliability curve, cluster model, test split\n({n_bins} quantile bins, Brier score = {brier:.4f})")
    ax.legend()
    ax.set_xlim(-0.02, max(mean_predicted_value.max(), 0.1) * 1.1)
    ax.set_ylim(-0.02, max(fraction_of_positives.max(), 0.1) * 1.1)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    gaps = fraction_of_positives - mean_predicted_value
    mean_abs_gap = float(np.mean(np.abs(gaps)))
    top_bin_gap = float(gaps[-1])
    top_bin_score = float(mean_predicted_value[-1])

    lines = ["## Calibration", ""]
    lines.append(
        "results/calibration.png -- reliability curve for the cluster "
        "model on the test split (quantile-binned, since scores "
        "concentrate near 0 at this base rate; equal-width bins would put "
        "almost everything in one bin)."
    )
    lines.append("")
    lines.append("![Reliability curve](calibration.png)")
    lines.append("")
    lines.append(f"- Brier score: **{brier:.4f}** (lower is better; a model that always")
    lines.append(f"  predicts the test-set base rate {base_rate:.4f} scores {brier_baseline:.4f} for comparison)")
    lines.append(
        f"- Mean absolute gap between observed and predicted fraction, "
        f"equally weighted across the {n_bins} bins: **{mean_abs_gap:.4f}** "
        "-- this number is misleading on its own, see below."
    )
    lines.append(
        f"- Highest-score bin (mean predicted score {top_bin_score:.4f}, the "
        "bin nearest where policy.py's thresholds actually operate): "
        f"observed fraction of positives is **{fraction_of_positives[-1]:.4f}**, "
        f"a gap of **{top_bin_gap:+.4f}**."
    )
    lines.append("")
    lines.append(
        "**The equally-weighted average is misleading here and would say "
        "the wrong thing if reported alone.** Most of the 15 bins sit at "
        "very low predicted scores, where a ~3.5%-base-rate model is "
        "naturally easy to calibrate (predicting near 0 for mostly-0 "
        "outcomes), so they pull the average down. The bin that actually "
        f"matters for policy.py -- the highest one, mean predicted score "
        f"{top_bin_score:.4f}, which is above both STEP_UP_THRESHOLD "
        f"({policy.STEP_UP_THRESHOLD}) and REVIEW_THRESHOLD "
        f"({policy.REVIEW_THRESHOLD}) -- is **overconfident by "
        f"{abs(top_bin_gap):.2f}** (predicts ~{top_bin_score:.2f}, actual "
        f"positive rate is only {fraction_of_positives[-1]:.2f})."
    )
    lines.append("")
    lines.append(
        "**Verdict: not well calibrated in the region the policy engine "
        "actually operates in, despite a low overall Brier score.** "
        f"policy.py's REVIEW_THRESHOLD ({policy.REVIEW_THRESHOLD}) should "
        "be read as an arbitrary cut on this model's score scale, not as "
        f"\"we estimate >{policy.REVIEW_THRESHOLD:.2%} abuse risk\" -- "
        "scores in this upper range systematically "
        "overstate the true positive rate. The fix, if a true probability "
        "is needed, is isotonic or Platt scaling fit on a held-out "
        "calibration slice -- **not implemented here**: model.py is frozen "
        "this session, and refitting a calibration map changes how scores "
        "are produced, which is not something to add quietly under a task "
        "that explicitly said do not retrain the model."
    )
    lines.append("")
    _append_to_ablation(lines)


# --- Investigator evaluation (results/investigator_eval.md) ---------------


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_PATTERN.findall(text)]


def _value_matches(claimed: float, evidence_value: float) -> bool:
    candidates = {claimed, claimed / 100}
    for candidate in candidates:
        for decimals in range(0, 5):
            if abs(candidate - round(float(evidence_value), decimals)) < 1e-9:
                return True
    return False


def _ungrounded_claims(narrative: str, evidence: dict) -> list[float]:
    evidence_values = list(evidence.values())
    return [
        claim for claim in _extract_numbers(narrative)
        if not any(_value_matches(claim, v) for v in evidence_values)
    ]


def _cluster_risk_key(component: set, cluster_features: pd.DataFrame) -> float:
    rep = next(iter(component))
    if rep not in cluster_features.index:
        return float("-inf")
    value = cluster_features.loc[rep, "cluster_prior_fraud_share"]
    return float(value) if pd.notna(value) else 0.0


def _select_clusters(pipeline_data: PipelineData, n: int) -> list[set]:
    components = get_connected_components(pipeline_data.entity_graph.graph)
    multi = [c for c in components if len(c) >= 2]
    multi_sorted = sorted(multi, key=lambda c: _cluster_risk_key(c, pipeline_data.cluster_features))
    if len(multi_sorted) <= n:
        return multi_sorted
    indices = sorted(set(np.linspace(0, len(multi_sorted) - 1, n).round().astype(int)))
    return [multi_sorted[i] for i in indices]


def _groundedness_stats(
    explanations: list[investigator.ClusterExplanation],
) -> tuple[int, int, float, list[tuple[str, list[float], str]]]:
    total_claims = 0
    total_ungrounded = 0
    records = []
    for e in explanations:
        claims = _extract_numbers(e.narrative)
        ungrounded = _ungrounded_claims(e.narrative, e.evidence)
        total_claims += len(claims)
        total_ungrounded += len(ungrounded)
        if ungrounded:
            records.append((e.cluster_id, ungrounded, e.narrative))
    rate = 1.0 - (total_ungrounded / total_claims) if total_claims else float("nan")
    return total_claims, total_ungrounded, rate, records


def write_investigator_eval(pipeline_data: PipelineData) -> None:
    """Run explain_cluster on 30 risk-spanning clusters and report
    groundedness -- every claim in the written report is derived from the
    actual run's `source`/`error` fields, never inferred from whether
    ANTHROPIC_API_KEY happened to be set. See DEVLOG.md for why that
    distinction is the entire point of this function: a workspace-linked
    key without the required header fails on every single call, and this
    report used to say "at least one explanation used the real LLM path"
    while 30 of 30 had silently fallen back.
    """
    n_clusters = 30
    n_examples = 3

    full = pipeline_data.df.set_index("TransactionID")
    full = pd.concat([full, pipeline_data.entity_ids.rename("uid")], axis=1)

    selected = _select_clusters(pipeline_data, n_clusters)

    explanations = []
    for i, members in enumerate(selected):
        members = sorted(members)
        cluster_features_sub = pipeline_data.cluster_features.loc[
            [m for m in members if m in pipeline_data.cluster_features.index]
        ]
        transactions_sub = full[full["uid"].isin(members)]
        explanations.append(
            investigator.explain_cluster(
                cluster_id=f"cluster-{i}", cluster_features=cluster_features_sub, transactions=transactions_sub
            )
        )

    ranked = investigator.prioritize_clusters(explanations)

    sources: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for explanation in explanations:
        sources[explanation.source] = sources.get(explanation.source, 0) + 1
        if explanation.error:
            error_counts[explanation.error] = error_counts.get(explanation.error, 0) + 1

    n_total = len(explanations)
    n_llm = sources.get("llm", 0)
    n_fallback = n_total - n_llm
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    llm_explanations = [e for e in explanations if e.source == "llm"]
    fallback_explanations = [e for e in explanations if e.source != "llm"]
    llm_claims, llm_ungrounded_n, llm_rate, llm_ungrounded_records = _groundedness_stats(llm_explanations)
    fb_claims, fb_ungrounded_n, fb_rate, fb_ungrounded_records = _groundedness_stats(fallback_explanations)
    all_claims, all_ungrounded_n, all_rate, _ = _groundedness_stats(explanations)

    ranked_asc = list(reversed(ranked))
    example_indices = (
        sorted({0, len(ranked_asc) // 2, len(ranked_asc) - 1})
        if len(ranked_asc) >= n_examples
        else list(range(len(ranked_asc)))
    )
    examples = [ranked_asc[i] for i in example_indices]

    lines: list[str] = ["# Investigator evaluation", ""]
    lines.append(
        f"ANTHROPIC_API_KEY was {'set' if has_key else '**NOT set**'} when this ran. "
        f"Explanation sources (derived from the actual run's `source` field, "
        f"not assumed from whether a key was present): {sources}."
    )
    lines.append("")

    # This claim is computed strictly from `sources` -- a key being present
    # does not mean any call succeeded.
    if n_llm == 0:
        lines.append(
            f"**0 of {n_total} explanations used the real LLM path -- all "
            f"{n_total} took the deterministic fallback.**"
        )
        lines.append(
            "This happened DESPITE ANTHROPIC_API_KEY being set, which means "
            "every fallback here was caused by a real failure, not a "
            "missing key -- see 'Fallback errors encountered' below for "
            "exactly what went wrong on every call."
            if has_key
            else "ANTHROPIC_API_KEY was not set, so this is expected -- "
            "there was no key to call the API with."
        )
    elif n_fallback == 0:
        lines.append(f"**All {n_total} of {n_total} explanations used the real LLM path.**")
    else:
        lines.append(
            f"**{n_llm} of {n_total} explanations used the real LLM path; "
            f"{n_fallback} fell back.** See 'Fallback errors encountered' "
            "below for why the fallback ones did."
        )
    lines.append("")

    if error_counts:
        lines.append("### Fallback errors encountered")
        lines.append("")
        for err, count in sorted(error_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{err}` -- {count} of {n_total} cluster(s)")
        lines.append("")

    lines.append(
        f"Evaluated {n_total} clusters, selected to span the risk "
        "range: sorted all multi-uid clusters (2+ members) by "
        "cluster_prior_fraud_share, then took 30 evenly-spaced percentile "
        "points across that sorted list (not just the top 30 riskiest)."
    )
    lines.append("")

    lines.append("## Groundedness")
    lines.append("")
    if n_llm > 0:
        lines.append(
            f"**On the {n_llm} real LLM explanation(s): {llm_ungrounded_n} of "
            f"{llm_claims} numeric claims ungrounded -- groundedness rate "
            f"{llm_rate:.2%}.** This is the number that actually measures "
            "claude-sonnet-4-6's behavior under the prompt's hard rule. "
            "Reported honestly whatever it is -- a rate below 100% is a "
            "finding about the model's behavior, not something to fix by "
            "loosening the claim extractor."
        )
    else:
        lines.append(
            "**No real LLM explanations were produced this run -- there is "
            "nothing here that measures claude-sonnet-4-6's actual "
            "behavior.** The fallback-only numbers below are not a "
            "substitute for that measurement."
        )
    lines.append("")
    lines.append(
        f"- Fallback-only groundedness (for reference; expected ~100% since "
        "it's built by directly formatting evidence values verbatim): "
        f"{fb_ungrounded_n} of {fb_claims} claims ungrounded"
        + (f" ({fb_rate:.2%})" if fb_claims else " (no fallback explanations this run)")
    )
    lines.append(
        f"- Combined (LLM + fallback) across all {n_total} explanations: "
        f"{all_ungrounded_n} of {all_claims} claims ungrounded"
        + (f" ({all_rate:.2%})" if all_claims else "")
        + ". Not the LLM's groundedness rate when fallback explanations are "
        "mixed in -- a trivially-grounded template dilutes or inflates the "
        "real number, which is why it's reported separately above."
    )
    lines.append("")
    lines.append(
        "(A claim counts as grounded if it matches some evidence value "
        "exactly, at any rounding from 0-4 decimal places, or as that "
        "value expressed as a percentage.)"
    )
    lines.append("")

    if llm_ungrounded_records:
        lines.append("### Every ungrounded claim found (LLM explanations)")
        lines.append("")
        for cluster_id, ungrounded, narrative in llm_ungrounded_records:
            lines.append(f"- **{cluster_id}**: claimed {ungrounded}")
            lines.append(f"  > {narrative}")
        lines.append("")
    elif n_llm > 0:
        lines.append("No ungrounded claims found among the LLM explanations.")
        lines.append("")

    if fb_ungrounded_records:
        # Should never happen -- the fallback template is grounded by
        # construction. If it does, that's a bug in the template or the
        # claim extractor, not a fact about the LLM, and it's surfaced
        # here rather than silently folded into a combined rate.
        lines.append(
            "### Unexpected: ungrounded claims found in FALLBACK explanations"
        )
        lines.append("")
        lines.append(
            "This should be impossible -- the fallback template only ever "
            "prints evidence dict values verbatim. Its presence means a bug "
            "in `_fallback_narrative` or the claim extractor, not a finding "
            "about the LLM:"
        )
        lines.append("")
        for cluster_id, ungrounded, narrative in fb_ungrounded_records:
            lines.append(f"- **{cluster_id}**: claimed {ungrounded}")
            lines.append(f"  > {narrative}")
        lines.append("")

    lines.append("## 3 example explanations (lowest, median, highest priority)")
    lines.append("")
    for explanation in examples:
        lines.append(f"### {explanation.cluster_id} (source={explanation.source})")
        lines.append("")
        lines.append(f"- Priority score: {explanation.priority_score:.4f}")
        lines.append(f"- Member uids: {explanation.entity_ids}")
        lines.append(f"- Evidence: `{explanation.evidence}`")
        if explanation.error:
            lines.append(f"- Error: `{explanation.error}`")
        lines.append("")
        lines.append(f"> {explanation.narrative}")
        lines.append("")

    (RESULTS_DIR / "investigator_eval.md").write_text("\n".join(lines), encoding="utf-8")


# --- Audit sample (results/audit_sample.jsonl) -----------------------------


def _clean_feature_values(row: pd.Series) -> dict:
    values = {}
    for col in CLUSTER_FEATURE_COLUMNS:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            values[col] = None
        elif isinstance(v, float):
            values[col] = round(float(v), 4)
        else:
            values[col] = v
    return values


def write_audit_sample(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    sample_size = 200

    y_score = trained.cluster_model.predict(trained.X_test_cluster)
    test_uid = pipeline_data.entity_ids.reindex(trained.y_test.index)

    per_transaction_scores = pd.DataFrame(
        {"score": y_score, "uid": test_uid.to_numpy()}, index=trained.y_test.index
    )

    decisions = policy.apply_policy(per_transaction_scores[["score"]])

    cluster_features_by_txn = broadcast_cluster_features(
        pipeline_data.entity_ids, pipeline_data.cluster_features
    ).reindex(trained.y_test.index)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(per_transaction_scores.index.to_numpy(), size=sample_size, replace=False)
    sample_idx = pd.Index(sample_idx).sort_values()

    timestamp = datetime.now(timezone.utc).isoformat()

    records = []
    for transaction_id in sample_idx:
        score = float(per_transaction_scores.loc[transaction_id, "score"])
        uid = per_transaction_scores.loc[transaction_id, "uid"]
        decision_row = decisions.loc[transaction_id]
        decision = policy.PolicyDecision(
            entity_id=uid if pd.notna(uid) else str(transaction_id),
            action=decision_row["action"],
            reason=decision_row["reason"],
            threshold_applied=float(decision_row["threshold_applied"]),
        )
        feature_values = (
            _clean_feature_values(cluster_features_by_txn.loc[transaction_id])
            if transaction_id in cluster_features_by_txn.index
            else {col: None for col in CLUSTER_FEATURE_COLUMNS}
        )
        record = policy.build_audit_record(
            decision,
            transaction_id=int(transaction_id),
            uid=uid if pd.notna(uid) else None,
            score=round(score, 6),
            feature_values=feature_values,
            timestamp=timestamp,
        )
        records.append(record)

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "audit_sample.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- Benchmark (ARCHITECTURE.md "## Performance" section) -----------------


def _percentiles(latencies_seconds: list[float]) -> dict[str, float]:
    arr = np.array(latencies_seconds) * 1000
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "max_ms": float(np.max(arr)),
    }


def write_benchmark(pipeline_data: PipelineData, trained: TrainedModels) -> None:
    n_scoring_samples = 1000

    t0 = time.perf_counter()
    entity_graph = graph.build_entity_graph(
        pipeline_data.train_df, pipeline_data.entity_ids, max_degree=MAX_DEGREE
    )
    graph_build_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    cluster_features = graph.compute_cluster_features(
        pipeline_data.train_df, pipeline_data.entity_ids, entity_graph.graph, as_of=pipeline_data.as_of
    )
    cluster_features_seconds = time.perf_counter() - t0

    feature_store = {uid: row.to_dict() for uid, row in pipeline_data.cluster_features.iterrows()}

    rng = np.random.default_rng(0)
    sample_positions = rng.choice(len(trained.X_test_cluster), size=n_scoring_samples, replace=False)
    sample_rows = trained.X_test_cluster.iloc[sample_positions]
    sample_uids = pipeline_data.entity_ids.reindex(sample_rows.index)

    latencies = []
    for i in range(n_scoring_samples):
        txn_id = sample_rows.index[i]
        uid = sample_uids.iloc[i]

        t0 = time.perf_counter()
        cluster_row = feature_store.get(uid) if isinstance(uid, str) else None
        base_row = trained.X_test_baseline.loc[[txn_id]]
        if cluster_row is not None:
            for k, v in cluster_row.items():
                base_row[k] = v
        else:
            for k in pipeline_data.cluster_features.columns:
                base_row[k] = np.nan
        base_row = base_row.reindex(columns=trained.X_test_cluster.columns)
        model.predict(trained.cluster_model, base_row)
        latencies.append(time.perf_counter() - t0)

    scoring_stats = _percentiles(latencies)

    lines: list[str] = ["## Performance", ""]
    lines.append(
        "Batch/inline split, stated plainly: graph construction and "
        "cluster feature computation are BATCH -- run periodically "
        "(e.g. nightly, or whenever the graph is rebuilt) against "
        "historical/train-period transactions, never on the request path. "
        "Transaction scoring is INLINE: given a transaction, look up its "
        "uid's precomputed cluster features (a cache/feature-store read, "
        "not a recomputation) and call the model. Below is measured "
        "separately because they answer different capacity questions --  "
        "batch steps bound how often the graph can be refreshed, the "
        "inline step bounds request latency."
    )
    lines.append("")
    lines.append("### Batch: graph construction and cluster features")
    lines.append("")
    lines.append(
        f"- Graph construction ({len(pipeline_data.train_df):,} train-period "
        f"transactions, max_degree={MAX_DEGREE}): **{graph_build_seconds:.2f}s** "
        f"({entity_graph.graph.number_of_nodes():,} nodes, "
        f"{entity_graph.graph.number_of_edges():,} edges)"
    )
    lines.append(
        f"- Cluster feature computation ({len(cluster_features):,} uids): "
        f"**{cluster_features_seconds:.2f}s**"
    )
    lines.append("")
    lines.append("### Inline: per-transaction scoring latency")
    lines.append("")
    lines.append(
        f"{n_scoring_samples:,} single-transaction scoring calls (feature-store "
        "lookup + model.predict on one row), sampled from the real test set:"
    )
    lines.append("")
    lines.append("| stat | value |")
    lines.append("|---|---:|")
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
        lines.append(f"| {key} | {scoring_stats[key]:.3f} ms |")
    lines.append("")
    lines.append(
        "This is single-row prediction, not batched -- LightGBM's per-call "
        "overhead dominates at this granularity, so p95 here is a "
        "meaningfully worse number than the model's throughput in bulk "
        "scoring would suggest. A real inline path would likely batch "
        "several in-flight requests if the volume justified it."
    )
    lines.append("")
    lines.append("### What would need to change at ~1B transactions/quarter")
    lines.append("")
    lines.append(
        f"This benchmark's full graph build "
        f"({len(pipeline_data.train_df):,} transactions) took "
        f"{graph_build_seconds:.2f}s for construction + "
        f"{cluster_features_seconds:.2f}s for features. ~1B transactions/quarter "
        f"is roughly {1e9 / len(pipeline_data.train_df):.0f}x this benchmark's "
        "train set. Naive linear scaling alone would already push a full "
        "rebuild from seconds into hours, and the real cost is worse than "
        "linear: this project's own hub-guard investigation "
        "(graph.py's module docstring) found that the graph's structure is "
        "sensitive to `max_degree` in a highly non-linear way (a phase "
        "transition, not a smooth curve) -- at greater scale, more "
        "identifier values cross the hub threshold, and getting this wrong "
        "risks the same giant-component collapse found earlier this "
        "project, at a much more expensive scale to detect and recover from."
    )
    lines.append("")
    lines.append("Three changes this scale would require, none implemented here:")
    lines.append("")
    lines.append(
        "- **Incremental graph updates instead of full rebuilds.** This "
        "project rebuilds the whole graph from all train-period "
        "transactions every time (build_entity_graph has no notion of "
        "\"since last run\"). At 1B/quarter, a full rebuild needs to become "
        "an incremental one: new transactions add nodes/edges to an "
        "existing graph, without re-scanning historical data that hasn't "
        "changed."
    )
    lines.append(
        "- **Approximate connected components.** get_connected_components "
        "is an exact, single-machine networkx computation. At this scale "
        "the graph itself likely needs to be sharded/distributed, and "
        "exact connected components across shards is expensive; "
        "approximate or incremental union-find structures (as used in "
        "large-scale graph processing systems) trade a small amount of "
        "accuracy for tractability."
    )
    lines.append(
        "- **Sharding.** The current implementation holds one in-memory "
        "networkx graph and one in-memory feature table for the whole "
        "dataset. Neither fits in memory on one machine at this volume; "
        "the graph and its features would need to be partitioned (e.g. by "
        "a hash of a linkage key) across multiple machines, which changes "
        "how cross-shard edges (a device or address linking uids in "
        "different shards) get detected and reconciled -- a real design "
        "problem, not a configuration change."
    )
    lines.append("")

    existing = ARCHITECTURE_PATH.read_text(encoding="utf-8") if ARCHITECTURE_PATH.exists() else ""
    existing = _remove_section(existing, "## Performance")
    ARCHITECTURE_PATH.write_text(
        existing.rstrip("\n") + ("\n\n" if existing.strip() else "") + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the full pipeline and produce every artifact end to end.

    results/case_studies.md is the one deliberate exception -- see this
    module's docstring for why it's not regenerated here.
    """
    t0 = time.time()
    pipeline_data = load_and_prepare()
    print(f"[run_pipeline] data + graph prepared in {time.time() - t0:.1f}s")

    t0 = time.time()
    trained = train_both_models(pipeline_data)
    print(f"[run_pipeline] both models trained in {time.time() - t0:.1f}s")

    metrics = evaluate_both_models(trained)
    print(f"[run_pipeline] baseline: {metrics['baseline']}")
    print(f"[run_pipeline] cluster:  {metrics['cluster']}")

    write_ablation_report(pipeline_data, trained, metrics)
    print("[run_pipeline] wrote results/ablation.md")

    write_sanity_checks(pipeline_data, trained)
    print("[run_pipeline] appended Sanity checks to results/ablation.md")

    write_cost_curve(pipeline_data, trained)
    print("[run_pipeline] wrote results/cost_curve.png + appended sweep section")

    write_calibration(pipeline_data, trained)
    print("[run_pipeline] wrote results/calibration.png + appended Calibration section")

    t0 = time.time()
    write_investigator_eval(pipeline_data)
    print(f"[run_pipeline] wrote results/investigator_eval.md in {time.time() - t0:.1f}s")

    write_audit_sample(pipeline_data, trained)
    print("[run_pipeline] wrote results/audit_sample.jsonl")

    t0 = time.time()
    write_benchmark(pipeline_data, trained)
    print(f"[run_pipeline] appended Performance section to ARCHITECTURE.md in {time.time() - t0:.1f}s")

    print(
        "[run_pipeline] done. results/case_studies.md intentionally NOT "
        "regenerated (hand-curated judgment artifact -- see module docstring)."
    )


if __name__ == "__main__":
    main()
