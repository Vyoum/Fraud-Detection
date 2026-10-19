"""python -m src.streaming.export --output artifacts/demo"""

import argparse
from pathlib import Path

from src import run_pipeline
from src.streaming.artifacts import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-parquet", help="Joined labeled history, local path or hdfs:// URI")
    args = parser.parse_args()
    pipeline = run_pipeline.load_and_prepare(history_parquet=args.history_parquet)
    trained = run_pipeline.train_both_models(pipeline)
    manifest = export_bundle(pipeline, trained, args.output)
    print(f"Exported {args.output}; model {manifest['model_version'][:12]}; "
          f"serve TransactionDT >= {manifest['valid_from_transaction_dt']}")


if __name__ == "__main__":
    main()
