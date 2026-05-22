"""Search low-FPR thresholds and summarize private CV predictions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def compute_binary_metrics(
    targets: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for target, probability in zip(targets, probabilities):
        pred = int(float(probability) >= threshold)
        target = int(bool(target))
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    beta = 0.5
    f05 = (1 + beta * beta) * precision * recall / ((beta * beta * precision) + recall + 1e-8)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr = fp / (fp + tn + 1e-8)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f05,
        "accuracy": accuracy,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive": sum(int(bool(x)) for x in targets),
        "negative": len(targets) - sum(int(bool(x)) for x in targets),
    }


def make_thresholds(start: float, end: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= end + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def select_threshold(
    targets: Sequence[int],
    probabilities: Sequence[float],
    max_fpr: float = 0.05,
    thresholds: Iterable[float] | None = None,
) -> Dict:
    threshold_values = list(thresholds or make_thresholds(0.01, 0.99, 0.01))
    metrics = [
        compute_binary_metrics(targets, probabilities, threshold)
        for threshold in threshold_values
    ]
    valid = [item for item in metrics if item["fpr"] <= max_fpr]

    if valid:
        best = max(
            valid,
            key=lambda item: (
                item["recall"],
                item["f0_5"],
                item["precision"],
                -item["fpr"],
                -item["threshold"],
            ),
        )
        best = dict(best)
        best["meets_fpr_constraint"] = True
    else:
        best = min(
            metrics,
            key=lambda item: (
                item["fpr"],
                -item["recall"],
                -item["f0_5"],
                -item["precision"],
            ),
        )
        best = dict(best)
        best["meets_fpr_constraint"] = False
    best["max_fpr"] = max_fpr
    return best


def read_predictions(path: Path) -> Tuple[List[int], List[float], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    targets = [int(float(row["true_label"])) for row in rows]
    probabilities = [float(row["pred_prob"]) for row in rows]
    return targets, probabilities, rows


def find_prediction_files(pred_dir: Path) -> List[Path]:
    files = sorted(pred_dir.glob("fold_*/predictions.csv"))
    if files:
        return files
    single = pred_dir / "predictions.csv"
    return [single] if single.exists() else []


def format_metric_line(name: str, metrics: Dict) -> str:
    return (
        f"{name}: threshold={metrics['threshold']:.4f}, "
        f"Precision={metrics['precision'] * 100:.1f}%, "
        f"Recall={metrics['recall'] * 100:.1f}%, "
        f"F1={metrics['f1'] * 100:.1f}%, "
        f"F0.5={metrics['f0_5'] * 100:.1f}%, "
        f"FPR={metrics['fpr'] * 100:.1f}%, "
        f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}"
    )


def summarize_private_cv(args) -> Dict:
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = find_prediction_files(pred_dir)
    if not prediction_files:
        raise FileNotFoundError(f"no predictions.csv found under {pred_dir}")

    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    fold_summaries = []
    all_targets: List[int] = []
    all_probabilities: List[float] = []

    for path in prediction_files:
        targets, probabilities, _ = read_predictions(path)
        fold_name = path.parent.name if path.parent != pred_dir else "all"
        best = select_threshold(targets, probabilities, args.max_fpr, thresholds)
        best["fold"] = fold_name
        best["prediction_file"] = str(path)
        fold_summaries.append(best)
        all_targets.extend(targets)
        all_probabilities.extend(probabilities)

    recommended_threshold = statistics.median([item["threshold"] for item in fold_summaries])
    overall_at_recommended = compute_binary_metrics(
        all_targets,
        all_probabilities,
        recommended_threshold,
    )
    overall_best = select_threshold(all_targets, all_probabilities, args.max_fpr, thresholds)

    summary = {
        "pred_dir": str(pred_dir),
        "output_dir": str(output_dir),
        "max_fpr": args.max_fpr,
        "threshold_search": {
            "start": args.threshold_start,
            "end": args.threshold_end,
            "step": args.threshold_step,
        },
        "recommended_threshold": recommended_threshold,
        "folds": fold_summaries,
        "overall_at_recommended_threshold": overall_at_recommended,
        "overall_best_threshold": overall_best,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "Private 5-fold CV summary",
        "=" * 60,
        f"prediction_files: {len(prediction_files)}",
        f"max_fpr_constraint: {args.max_fpr:.4f}",
        f"recommended_threshold_median: {recommended_threshold:.4f}",
        "",
        format_metric_line("overall@recommended", overall_at_recommended),
        format_metric_line("overall@best", overall_best),
        "",
        "Per fold",
        "-" * 60,
    ]
    for item in fold_summaries:
        lines.append(format_metric_line(item["fold"], item))

    report = "\n".join(lines) + "\n"
    with open(output_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize private CV predictions")
    parser.add_argument("--pred-dir", default="experiments/private_cv")
    parser.add_argument("--output-dir", default="experiments/private_cv")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--threshold-start", type=float, default=0.01)
    parser.add_argument("--threshold-end", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args(argv)
    summarize_private_cv(args)


if __name__ == "__main__":
    main()
