"""Load and join the raw IEEE-CIS transaction and identity files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRANSACTION_FILENAME = "train_transaction.csv"
IDENTITY_FILENAME = "train_identity.csv"

_CATEGORY_MAX_UNIQUE = 50


def load_history_parquet(path: str) -> pd.DataFrame:
    """Read joined, labeled history from local Parquet or HDFS via PyArrow.

    HDFS requires libhdfs and a Hadoop client configuration. The raw Kafka
    archive has an envelope schema and must first be flattened and joined
    with observed labels by jobs/prepare_history.py.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1", "isFraud"}
    if missing := required - set(df.columns):
        raise ValueError(f"Historical Parquet is missing columns: {sorted(missing)}")
    if df["TransactionID"].duplicated().any():
        raise ValueError("Historical Parquet must have one row per TransactionID")
    if df["TransactionDT"].isna().any() or not df["isFraud"].isin([0, 1]).all():
        raise ValueError("History requires valid timestamps and explicit binary labels")
    return _downcast_dtypes(df)


def load_transactions(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load train_transaction.csv, left-joined with train_identity.csv.

    Uses a left join (not inner): train_identity.csv covers only ~24% of
    TransactionIDs, so an inner join would silently discard the other ~76%
    of rows. Identity columns are NaN for transactions with no identity
    record.

    Args:
        path: Directory containing train_transaction.csv and
            train_identity.csv (e.g. the data/ directory produced by
            scripts/download_data.sh).
        nrows: If given, only read this many rows from train_transaction.csv.
            Useful for fast local iteration; leave as None for the full file.

    Returns:
        One row per transaction (TransactionID remains a column), with
        float64 columns downcast to float32 and low-cardinality object
        columns (at most _CATEGORY_MAX_UNIQUE distinct values) cast to
        category, to keep the ~590k-row by ~430-column file from ballooning
        to several GB in memory.

    Raises:
        FileNotFoundError: If either CSV is missing from `path`.
    """
    path = Path(path)
    transaction_path = path / TRANSACTION_FILENAME
    identity_path = path / IDENTITY_FILENAME

    missing = [p for p in (transaction_path, identity_path) if not p.exists()]
    if missing:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing raw data file(s): {missing_names} in '{path}'. "
            "Run scripts/download_data.sh to fetch the IEEE-CIS dataset "
            "from Kaggle before loading transactions."
        )

    transactions = pd.read_csv(transaction_path, nrows=nrows)
    identity = pd.read_csv(identity_path)

    before_bytes = (
        transactions.memory_usage(deep=True).sum()
        + identity.memory_usage(deep=True).sum()
    )

    merged = transactions.merge(identity, on="TransactionID", how="left")
    merged = _downcast_dtypes(merged)

    after_bytes = merged.memory_usage(deep=True).sum()

    print(f"[data] loaded {len(merged):,} rows, {merged.shape[1]} columns")
    print(f"[data] memory before downcast: {before_bytes / 1e6:.1f} MB")
    print(f"[data] memory after downcast:  {after_bytes / 1e6:.1f} MB")

    return merged


def _downcast_dtypes(
    df: pd.DataFrame, category_max_unique: int = _CATEGORY_MAX_UNIQUE
) -> pd.DataFrame:
    """Downcast float64 -> float32 and low-cardinality object -> category.

    Args:
        df: DataFrame to downcast.
        category_max_unique: An object column is cast to category only if it
            has at most this many distinct non-null values; higher-cardinality
            object columns (e.g. DeviceInfo) are left as-is.

    Returns:
        The same DataFrame with dtypes downcast.
    """
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique(dropna=True) <= category_max_unique:
            df[col] = df[col].astype("category")

    return df
