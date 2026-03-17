#!/usr/bin/env python3
"""Plot task accuracy curves against interpolation alpha."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_PREFIX = "predict_exact_match_for_"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot interpolation accuracy curves from eval results.")
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root directory that contains evals/alpha_*/<eval_stage>/predict_results.json.",
    )
    parser.add_argument(
        "--eval-stage",
        default="4-agnews",
        help="Which eval stage to read, for example 4-agnews.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional explicit task list. If omitted, tasks are inferred from metric keys.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <results-root>/plots/<eval-stage>_alpha_accuracy.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom chart title.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def extract_alpha(path):
    for part in path.parts:
        if part.startswith("alpha_"):
            return float(part.removeprefix("alpha_"))
    raise ValueError(f"Could not infer alpha from {path}")


def infer_tasks(metrics):
    tasks = []
    for key in metrics:
        if not key.startswith(METRIC_PREFIX):
            continue
        task_name = key.removeprefix(METRIC_PREFIX)
        if task_name.isupper():
            continue
        tasks.append(task_name)
    return sorted(set(tasks))


def main():
    args = parse_args()
    results_root = Path(args.results_root).expanduser()
    if args.output is None:
        output_path = results_root / "plots" / f"{args.eval_stage}_alpha_accuracy.png"
    else:
        output_path = Path(args.output).expanduser()

    predict_files = sorted(results_root.glob(f"evals/alpha_*/{args.eval_stage}/predict_results.json"))
    if not predict_files:
        raise FileNotFoundError(
            f"No predict_results.json files found under {results_root / 'evals'} for stage {args.eval_stage}"
        )

    points = []
    task_names = list(args.tasks) if args.tasks else None
    for predict_file in predict_files:
        metrics = load_json(predict_file)
        alpha = extract_alpha(predict_file)
        if task_names is None:
            task_names = infer_tasks(metrics)
        points.append((alpha, metrics))

    if not task_names:
        raise ValueError("No task metrics found. Try passing --tasks explicitly.")

    points.sort(key=lambda item: item[0])

    plt.figure(figsize=(9, 5.5))
    for task_name in task_names:
        xs = []
        ys = []
        metric_key = f"{METRIC_PREFIX}{task_name}"
        for alpha, metrics in points:
            if metric_key not in metrics:
                continue
            xs.append(alpha)
            ys.append(metrics[metric_key])
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=task_name)

    plt.xlabel("alpha")
    plt.ylabel("exact match")
    plt.title(args.title or f"Interpolation Accuracy vs Alpha ({args.eval_stage})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
