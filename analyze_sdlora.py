
import argparse
import csv
import logging
import os
from typing import Dict, List, Tuple

import torch


def find_adapter_file(adapter_dir: str) -> Tuple[str]:
    pt = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(pt):
        return pt
    raise FileNotFoundError(f"adapter weights not found in {adapter_dir}")


def load_state_dict(adapter_dir: str) -> Dict:
    path = adapter_dir+'/adapter_model.bin'
    # torch.load can read safetensors via safe_open? For simplicity rely on safetensors when present using safetensors lib.
    return torch.load(path, map_location="cpu", weights_only=True)


def parse_scalings(sd: Dict, adapter_name: str = "default" , task_id=0) -> List[Dict]:
    """Extract historical_scalings entries to a list of rows: {layer, direction, task, value}."""
    rows: List[Dict] = []
    for k, v in sd.items():
        if "historical_scalings" not in k:
            continue
        # ...historical_scalings.dir_0
        parts = k.split("historical_scalings.")
        if len(parts) < 2:
            continue
        dir_key = parts[1] # dir_0
        # try:
        #     task = int(dir_key.split("_")[1])
        # except Exception:
        #     task = None
        layer = parts[0].rstrip(".")  # dotted layer path before historical_scalings
        try:
            val = float(v.item() if hasattr(v, "item") else float(v))
        except Exception:
            # if it's a tensor parameter
            try:
                val = float(getattr(v, "data", v))
            except Exception:
                continue
        rows.append({"layer": layer, "direction": dir_key, "task": task_id, "value": val})
    return rows


def auto_discover(root: str, pattern: str) -> List[str]:
    # for i in os.scandir(root):
    #     print(i)
    dirs = [f.name for f in os.scandir(root) if f.is_dir()]
    dirs = [os.path.join(root, d,'adapter') for d in sorted(dirs)]
    return dirs


def write_csv(rows: List[Dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "layer", "direction", "value"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "task": r.get("task"),
                "layer": r.get("layer"),
                "direction": r.get("direction"),
                "value": r.get("value"),
            })

import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
def main():
    parser = argparse.ArgumentParser(description="Analyze SDLoRA historical_scalings across tasks")
    parser.add_argument("--adapter-dirs", nargs="*", default=[], help="List of adapter directories to analyze, in task order")
    parser.add_argument("--root", default="logs_and_outputs/sdlora/order_1/outputs", help="Root to auto discover if adapter-dirs empty")
    parser.add_argument("--match", default="adapter", help="Substring to match when discovering adapter dirs")
    parser.add_argument("--adapter-name", default="default", help="Adapter name used when saving")
    parser.add_argument("--out-csv", default="analyze/sdlora.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Plot per-layer average scaling over tasks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    log = logging.getLogger("analyze_sdlora")

    adapter_dirs: List[str] = args.adapter_dirs
    if not adapter_dirs:
        adapter_dirs = auto_discover(args.root, args.match)
        log.info(f"Discovered {len(adapter_dirs)} adapter dirs")
    if not adapter_dirs:
        log.error("No adapter directories found")
        return

    all_rows: List[Dict] = []
    for idx, d in enumerate(adapter_dirs):
        sd = load_state_dict(d)
        rows = parse_scalings(sd, adapter_name=args.adapter_name,task_id=idx)
        # 若 state_dict 未显式包含任务号，从目录顺序补齐
        # for r in rows:
        #     if r.get("task") is None:
        #         r["task"] = idx
        all_rows.extend(rows)

    if not all_rows:
        log.warning("No historical_scalings found in provided adapters")
    write_csv(all_rows, args.out_csv)
    log.info(f"Wrote CSV: {args.out_csv} with {len(all_rows)} rows")

    if args.plot:
        import pandas as pd
        import matplotlib.pyplot as plt
        import re

        df = pd.DataFrame(all_rows)
        if df.empty:
            log.warning("Empty data; skip plotting")
            return

        # 按 (task, direction) 聚合，得到每个方向在各任务的平均 scaling（跨层平均）
        pivot_dir = df.groupby(["task", "direction"])["value"].mean().reset_index()

        # 对 direction 按数字排序（dir_0, dir_1, ...），若无法解析数字则按字典序
        def _dir_key(s: str):
            m = re.search(r"(\d+)", str(s))
            return (0, int(m.group(1))) if m else (1, str(s))
        directions = sorted(pivot_dir["direction"].dropna().unique().tolist(), key=_dir_key)

        # 任务范围与刻度（以 1 起始的可读标签）
        tasks_sorted = sorted(pivot_dir["task"].dropna().astype(int).unique().tolist())
        if not tasks_sorted:
            log.warning("No tasks found after grouping; skip plotting")
            return

        plt.figure(figsize=(10, 6))
        for dkey in directions:
            dfd = pivot_dir[pivot_dir["direction"] == dkey].copy()
            if dfd.empty:
                continue
            dfd["task"] = dfd["task"].astype(int)
            dfd = dfd.sort_values("task")
            plt.plot(
                dfd["task"],
                dfd["value"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=6,
                label=str(dkey),
            )

        plt.xlabel("Completed Tasks", fontsize=12)
        plt.ylabel("Historical scaling (avg over layers)", fontsize=12)
        plt.title("SD-LoRA: historical_scalings by direction (dir_i) over tasks", fontsize=14)
        # x 轴刻度与标签（Task 从 1 开始展示）
        plt.xticks(tasks_sorted, [f"Task {t+1}" for t in tasks_sorted])
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        out_png = os.path.splitext(args.out_csv)[0] + "_dirs.png"
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        log.info(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()


