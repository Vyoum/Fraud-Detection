"""Portable LightGBM bundles with exact training schema and historical features.

Export reuses the research pipeline. Its training-label and threshold-selection
limitations remain documented in docs/STREAMING.md; this does not claim a new
causal training experiment. Inference uses only a frozen pre-boundary snapshot.
"""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import entities, policy
from src.streaming.events import InvalidEvent, utc_now, validate_transaction


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(pipeline, trained, destination: Path) -> dict:
    destination = Path(destination)
    if pipeline.train_df["TransactionDT"].max() >= pipeline.as_of:
        raise ValueError("Feature history must be strictly before the serving boundary")
    # Refuse overwriting a deployed bundle. manifest.json is written last so
    # an interrupted export cannot be loaded as a complete bundle.
    destination.mkdir(parents=True, exist_ok=False)
    trained.cluster_model.save_model(str(destination / "model.txt"))
    pipeline.cluster_features.to_parquet(destination / "cluster_features.parquet")

    import networkx as nx

    membership = {
        uid: min(component)
        for component in nx.connected_components(pipeline.entity_graph.graph)
        for uid in component
    }
    pd.Series(membership, name="cluster_id").rename_axis("entity_id").to_frame().to_parquet(
        destination / "membership.parquet"
    )
    # The replay transport contains observations only; labels never leave here.
    pipeline.test_df.drop(columns=["isFraud", "uid"], errors="ignore").to_parquet(
        destination / "replay.parquet", index=False
    )
    cluster_columns = list(pipeline.cluster_features.columns)
    schema = {}
    for name, column in trained.X_train_cluster.items():
        if isinstance(column.dtype, pd.CategoricalDtype):
            schema[name] = {"kind": "category", "categories": column.cat.categories.tolist()}
        else:
            schema[name] = {"kind": "numeric", "dtype": str(column.dtype)}
    hashes = {name: file_hash(destination / name) for name in (
        "model.txt", "cluster_features.parquet", "membership.parquet", "replay.parquet"
    )}
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model_version": hashes["model.txt"],
        "feature_snapshot_version": hashes["cluster_features.parquet"],
        "valid_from_transaction_dt": float(pipeline.as_of),
        "history_max_transaction_dt": float(pipeline.train_df["TransactionDT"].max()),
        "feature_columns": list(trained.X_train_cluster.columns),
        "cluster_columns": cluster_columns,
        "input_columns": [c for c in schema if c not in cluster_columns],
        "schema": schema,
        "thresholds": dict(policy.DEFAULT_THRESHOLDS),
        "files": hashes,
        "training_method": "existing research pipeline; full train-window aggregates",
        "threshold_provenance": "existing illustrative thresholds selected on research test scores",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


class ScoringBundle:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m["schema_version"] != 1:
            raise ValueError("Unsupported bundle schema")
        for filename in ("model.txt", "cluster_features.parquet", "membership.parquet"):
            if file_hash(directory / filename) != m["files"][filename]:
                raise ValueError(f"Bundle checksum mismatch: {filename}")
        if m["history_max_transaction_dt"] >= m["valid_from_transaction_dt"]:
            raise ValueError("Non-causal feature snapshot")
        self.model = lgb.Booster(model_file=str(directory / "model.txt"))
        self.features = pd.read_parquet(directory / "cluster_features.parquet")
        self.membership = pd.read_parquet(directory / "membership.parquet")["cluster_id"]
        if self.model.feature_name() != m["feature_columns"]:
            raise ValueError("Model feature order does not match manifest")

    def prepare(self, event: dict) -> tuple[pd.DataFrame, str | None]:
        tx = validate_transaction(event)
        m = self.manifest
        if tx["TransactionDT"] < m["valid_from_transaction_dt"]:
            raise InvalidEvent("Transaction predates this model/feature snapshot; use an older bundle")
        required = set(m["input_columns"]) | {"TransactionID", "TransactionDT", "card1", "addr1", "D1"}
        if missing := required - tx.keys():
            raise InvalidEvent(f"Missing fields (send explicit nulls for unavailable values): {sorted(missing)}")
        allowed = set(m["input_columns"]) | {"TransactionID"}
        if extra := tx.keys() - allowed:
            raise InvalidEvent(f"Unexpected fields: {sorted(extra)}")
        for name in m["input_columns"]:
            value = tx[name]
            spec = m["schema"][name]
            if spec["kind"] == "numeric" and value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidEvent(f"{name}: expected a number or null")
        # A JSON null in a one-row frame otherwise becomes object(None), which
        # is not roundable by the existing batch UID resolver.
        identity_row = pd.DataFrame([tx])
        for name in ("card1", "addr1", "D1", "TransactionDT"):
            identity_row[name] = pd.to_numeric(identity_row[name], errors="raise").astype("float64")
        try:
            uid_value = entities.resolve_entities(identity_row).iloc[0]
        except (ValueError, TypeError, OverflowError) as exc:
            raise InvalidEvent("UID components cannot be represented by the entity resolver") from exc
        uid = str(uid_value) if pd.notna(uid_value) else None
        values = {name: tx.get(name) for name in m["input_columns"]}
        historical = self.features.loc[uid] if uid in self.features.index else None
        values.update({name: historical[name] if historical is not None else np.nan
                       for name in m["cluster_columns"]})
        X = pd.DataFrame([values], index=pd.Index([tx["TransactionID"]], name="TransactionID"))
        for name, spec in m["schema"].items():
            if spec["kind"] == "category":
                known = X[name].where(X[name].isin(spec["categories"]))
                X[name] = pd.Categorical(known, categories=spec["categories"])
            else:
                # Numeric float columns retain training precision; nullable int
                # inputs use float64 so LightGBM receives a real NaN.
                dtype = spec["dtype"] if spec["dtype"].startswith("float") else "float64"
                X[name] = X[name].astype(dtype)
        return X.reindex(columns=m["feature_columns"]), uid

    def score(self, event: dict) -> dict:
        X, uid = self.prepare(event)
        score = float(self.model.predict(X, num_threads=1)[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Model produced an invalid probability score")
        m = self.manifest
        tx = event["transaction"]
        decision = policy.decide(uid or str(tx["TransactionID"]), score, self.features,
                                 thresholds=m["thresholds"])
        feature_values = {name: None if pd.isna(X.iloc[0][name]) else float(X.iloc[0][name])
                          for name in m["cluster_columns"]}
        record = policy.build_audit_record(
            decision, tx["TransactionID"], uid, score, feature_values, utc_now(), m["model_version"]
        )
        identity = f"{event['event_id']}:{m['model_version']}:{m['feature_snapshot_version']}:{m['thresholds']}"
        record.update({
            "schema_version": 1,
            "decision_id": hashlib.sha256(identity.encode()).hexdigest(),
            "event_id": event["event_id"],
            "transaction_dt": tx["TransactionDT"],
            "cluster_id": self.membership.get(uid) if uid is not None else None,
            "feature_snapshot_version": m["feature_snapshot_version"],
            "snapshot_as_of": m["valid_from_transaction_dt"],
        })
        return record
