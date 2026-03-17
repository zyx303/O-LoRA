#!/usr/bin/env python3
"""Collect interpolation eval outputs into a single CSV."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize interpolation predict_results.json files.")
    parser.add_argument("--results-root", required=True, help="Root directory created by the interpolation run script.")
    parser.add_argument("--output", required=True, help="Path to the CSV summary to write.")
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def extract_alpha(path):
    for part in path.parts:
        if part.startswith("alpha_"):
            return part.removeprefix("alpha_")
    return ""


def main():
    args = parse_args()
    results_root = Path(args.results_root).expanduser()
    output_path = Path(args.output).expanduser()

    records = []
    for predict_path in sorted(results_root.glob("evals/alpha_*/4-agnews/predict_results.json")):
        alpha = extract_alpha(predict_path)
        metrics = load_json(predict_path)
        adapter_meta_path = results_root / "adapters" / f"alpha_{alpha}" / "interpolation_meta.json"
        adapter_meta = load_json(adapter_meta_path) if adapter_meta_path.exists() else {}

        record = {
            "alpha": alpha,
            "adapter_a": adapter_meta.get("adapter_a", ""),
            "adapter_b": adapter_meta.get("adapter_b", ""),
            "predict_exact_match": metrics.get("predict_exact_match", ""),
            "predict_rougeL": metrics.get("predict_rougeL", ""),
        }

        task_exact = []
        for key, value in metrics.items():
            if key.startswith("predict_exact_match_for_"):
                record[key] = value
                task_exact.append(value)
            elif key.startswith("predict_rougeL_for_"):
                record[key] = value

        if task_exact:
            record["mean_task_exact_match"] = round(sum(task_exact) / len(task_exact), 4)

        records.append(record)

    if not records:
        raise FileNotFoundError(f"No interpolation eval results found under {results_root}")

    fieldnames = sorted({field for record in records for field in record.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    for record in records:
        print(
            f"alpha={record['alpha']} "
            f"overall_EM={record.get('predict_exact_match', '')} "
            f"mean_task_EM={record.get('mean_task_exact_match', '')}"
        )
    print(f"summary_csv={output_path}")


if __name__ == "__main__":
    main()
