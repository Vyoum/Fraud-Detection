"""Serving parity and delivery invariants using small unit fixtures, no API/broker."""

from types import SimpleNamespace

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("confluent_kafka")

from src import entities, graph, model, policy
from src.data import load_history_parquet
from src.streaming.artifacts import ScoringBundle, export_bundle
from src.streaming.events import InvalidEvent, decode, encode
from src.streaming.producer import replay_events
from src.streaming.scorer import process_message
from src.streaming.store import open_store, read_decisions, save_decision


@pytest.fixture
def exported(tmp_path):
    # These are tiny schema/metric fixtures, not realistic abuse generation.
    train = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4], "TransactionDT": [0, 10, 20, 30],
        "card1": [1, 1, 2, 2], "addr1": [10.0, 10.0, 20.0, 20.0], "D1": [0.0] * 4,
        "TransactionAmt": pd.Series([1.1, 2.2, 3.3, 4.4], dtype="float32"),
        "ProductCD": pd.Categorical(["a", "b", "a", "b"], categories=["b", "a"]),
        "P_emaildomain": ["a", "a", "b", "b"], "isFraud": [0, 1, 0, 1],
    })
    test = train.iloc[[0, 1, 2]].copy()
    test["TransactionID"] = [5, 6, 7]
    test["TransactionDT"] = [40, 50, 60]
    test["ProductCD"] = ["b", "unseen", "a"]
    test.loc[test.index[-1], "addr1"] = np.nan
    uid = entities.resolve_entities(train)
    g = nx.Graph()
    g.add_nodes_from(uid.unique())
    cf = graph.compute_cluster_features(train, uid, g, as_of=40)
    train_cf = cf.reindex(uid.to_numpy())
    train_cf.index = uid.index
    X = model.build_feature_matrix(train, train_cf)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1,
                         "min_data_in_leaf": 1, "min_data_in_bin": 1, "seed": 42},
                        lgb.Dataset(X, label=train["isFraud"]), num_boost_round=5)
    pipeline = SimpleNamespace(train_df=train, test_df=test, as_of=40,
                               cluster_features=cf, entity_graph=SimpleNamespace(graph=g))
    trained = SimpleNamespace(cluster_model=booster, X_train_cluster=X)
    destination = tmp_path / "bundle"
    export_bundle(pipeline, trained, destination)
    return destination, pipeline, trained


def test_exported_model_matches_batch_scores_and_category_order(exported):
    path, pipeline, trained = exported
    bundle = ScoringBundle(path)
    for _, event in replay_events(path):
        tx = pd.DataFrame([event["transaction"]])
        for name in ("card1", "addr1", "D1", "TransactionDT"):
            tx[name] = pd.to_numeric(tx[name]).astype("float64")
        uid = entities.resolve_entities(tx)
        cf = pipeline.cluster_features.reindex(uid.to_numpy())
        cf.index = uid.index
        expected = model.build_feature_matrix(tx, cf)
        for name in trained.X_train_cluster.select_dtypes("category"):
            expected[name] = pd.Categorical(expected[name], categories=trained.X_train_cluster[name].cat.categories)
        for name in trained.X_train_cluster.select_dtypes("float32"):
            expected[name] = expected[name].astype("float32")
        expected = expected.reindex(columns=trained.X_train_cluster.columns)
        record = bundle.score(event)
        assert record["score"] == pytest.approx(float(trained.cluster_model.predict(expected, num_threads=1)[0]), abs=1e-12)
        assert record["action"] == policy.decide("x", record["score"], pd.DataFrame()).action
        assert record["model_version"] == bundle.manifest["model_version"]
        assert "isFraud" not in event["transaction"]
        assert decode(encode(record)) == record


def test_null_uid_retains_transaction_with_null_cluster_features(exported):
    path, _, _ = exported
    event = list(replay_events(path))[-1][1]
    record = ScoringBundle(path).score(event)
    assert record["uid"] is None
    assert record["cluster_id"] is None
    assert all(value is None for value in record["feature_values"].values())


@pytest.mark.parametrize("change, expected", [
    ({"TransactionDT": 39}, "predates"),
    ({"isFraud": 1}, "labels topic"),
    ({"cluster_prior_fraud_share": 1.0}, "Unexpected fields"),
    ({"TransactionAmt": "bad"}, "expected a number"),
    ({"TransactionDT": float("nan")}, "finite"),
])
def test_serving_rejects_leakage_and_malformed_inputs(exported, change, expected):
    path, _, _ = exported
    event = next(replay_events(path))[1]
    event["transaction"].update(change)
    with pytest.raises(InvalidEvent, match=expected):
        ScoringBundle(path).score(event)


def test_missing_field_must_be_explicit_null(exported):
    path, _, _ = exported
    event = next(replay_events(path))[1]
    del event["transaction"]["ProductCD"]
    with pytest.raises(InvalidEvent, match="Missing fields"):
        ScoringBundle(path).score(event)


def test_corrupt_bundle_is_not_loaded(exported):
    path, _, _ = exported
    with (path / "model.txt").open("a") as stream:
        stream.write("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ScoringBundle(path)


def test_export_cannot_overwrite_deployed_bundle(exported):
    path, pipeline, trained = exported
    with pytest.raises(FileExistsError):
        export_bundle(pipeline, trained, path)


def test_duplicate_decision_is_idempotent_across_restart(exported, tmp_path):
    path, _, _ = exported
    event = next(replay_events(path))[1]
    first = ScoringBundle(path).score(event)
    second = ScoringBundle(path).score(event)
    assert first["decision_id"] == second["decision_id"]
    db = tmp_path / "decisions.sqlite"
    connection = open_store(db)
    save_decision(connection, first)
    connection.close()
    connection = open_store(db)
    save_decision(connection, second)
    connection.close()
    counts, records = read_decisions(db)
    assert sum(counts.values()) == 1
    assert records == [first]


class FakeMessage:
    def __init__(self, raw):
        self.raw = raw

    def value(self): return self.raw
    def topic(self): return "fraud.transactions"
    def partition(self): return 0
    def offset(self): return 17


class FakeProducer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def begin_transaction(self): self.calls.append("begin")
    def produce(self, topic, **kwargs): self.calls.append((topic, kwargs))
    def send_offsets_to_transaction(self, offsets, metadata):
        self.calls.append(("offset", offsets[0].offset))
    def commit_transaction(self):
        if self.fail:
            raise RuntimeError("broker unavailable")
        self.calls.append("commit")
    def abort_transaction(self): self.calls.append("abort")


def test_output_and_offset_commit_are_atomic(exported):
    path, _, _ = exported
    event = next(replay_events(path))[1]
    producer = FakeProducer()
    consumer = SimpleNamespace(consumer_group_metadata=lambda: "group")
    process_message(FakeMessage(encode(event)), ScoringBundle(path), consumer, producer)
    assert producer.calls[0] == "begin"
    assert producer.calls[1][0] == "fraud.decisions"
    assert producer.calls[2:] == [("offset", 18), "commit"]


def test_transaction_failure_aborts_without_commit(exported):
    path, _, _ = exported
    producer = FakeProducer(fail=True)
    consumer = SimpleNamespace(consumer_group_metadata=lambda: "group")
    with pytest.raises(RuntimeError, match="broker unavailable"):
        process_message(FakeMessage(encode(next(replay_events(path))[1])), ScoringBundle(path), consumer, producer)
    assert producer.calls[-1] == "abort"
    assert "commit" not in producer.calls


def test_invalid_event_is_dead_lettered_and_offset_committed():
    producer = FakeProducer()
    consumer = SimpleNamespace(consumer_group_metadata=lambda: "group")
    process_message(FakeMessage(b"invalid-json"), None, consumer, producer)
    assert producer.calls[1][0] == "fraud.errors"
    assert producer.calls[-2:] == [("offset", 18), "commit"]


@pytest.mark.parametrize("raw", [b"[]", b"null", b'{"schema_version":1,"x":NaN}', b'\xff'])
def test_json_contract_rejects_invalid_values(raw):
    with pytest.raises(InvalidEvent):
        decode(raw)


def test_history_loader_refuses_unknown_labels_and_duplicates(exported, tmp_path):
    _, pipeline, _ = exported
    path = tmp_path / "history.parquet"
    frame = pipeline.train_df.copy()
    frame.to_parquet(path)
    assert len(load_history_parquet(str(path))) == len(frame)
    frame.loc[0, "isFraud"] = np.nan
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="binary labels"):
        load_history_parquet(str(path))
    frame = pd.concat([pipeline.train_df, pipeline.train_df])
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="one row"):
        load_history_parquet(str(path))


def test_live_dashboard_opens_without_raw_dataset_or_training(exported, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    path, _, _ = exported
    record = ScoringBundle(path).score(next(replay_events(path))[1])
    db = tmp_path / "live.sqlite"
    connection = open_store(db)
    save_decision(connection, record)
    connection.close()
    monkeypatch.setenv("FRAUD_DASHBOARD_MODE", "live")
    monkeypatch.setenv("FRAUD_DECISION_DB", str(db))
    # Failure would expose accidental eager research-pipeline loading.
    from src import run_pipeline
    monkeypatch.setattr(run_pipeline, "load_and_prepare", lambda: pytest.fail("Live mode loaded the research pipeline"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert len(app.exception) == 0
    assert app.metric[0].value == "1"
