"""Evaluation harness: score a run's event log against ground truth.

Ground truth is a CSV with one row per vehicle:
    id,start_s,end_s,wrong
    v1,0.0,4.2,yes
    v2,3.1,9.0,no

Predictions are the "violation" events from the JSONL log written by a
detection run. A violation matches a wrong-way ground-truth row when its
timestamp falls inside [start_s - tolerance, end_s + tolerance]. Outputs the
confusion matrix and the accuracy/precision/recall/F1 metrics comparable to
Table 4 of the paper.
"""

from __future__ import annotations

import argparse
import csv
import json

TRUTHY = {"yes", "y", "true", "1", "wrong"}


def load_ground_truth(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "wrong": row["wrong"].strip().lower() in TRUTHY,
            })
    return rows


def load_violations(path):
    events = []
    with open(path) as f:
        for line in f:
            event = json.loads(line)
            if event.get("kind") == "violation":
                events.append(event)
    return events


def evaluate(ground_truth, violations, tolerance=2.0):
    matched_events = set()
    tp = fn = 0
    false_alarms_on_negatives = 0

    def hits_for(row):
        lo, hi = row["start_s"] - tolerance, row["end_s"] + tolerance
        return [i for i, e in enumerate(violations)
                if i not in matched_events and lo <= e["t_s"] <= hi]

    # Match wrong-way rows first so a negative row's window can't steal a
    # violation that belongs to an overlapping wrong-way vehicle.
    for row in ground_truth:
        if not row["wrong"]:
            continue
        hits = hits_for(row)
        if hits:
            matched_events.add(hits[0])
            tp += 1
        else:
            fn += 1
    for row in ground_truth:
        if row["wrong"]:
            continue
        hits = hits_for(row)
        if hits:
            matched_events.add(hits[0])
            false_alarms_on_negatives += 1

    unmatched = len(violations) - len(matched_events)
    fp = false_alarms_on_negatives + unmatched
    tn = sum(1 for r in ground_truth if not r["wrong"]) - false_alarms_on_negatives

    total = tp + fp + fn + tn
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="result/events.jsonl",
                        help="JSONL event log produced by a detection run")
    parser.add_argument("--ground-truth", required=True,
                        help="Ground-truth CSV (id,start_s,end_s,wrong)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Seconds of slack when matching events to truth")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    violations = load_violations(args.events)
    metrics = evaluate(ground_truth, violations, args.tolerance)

    print(f"Ground truth: {len(ground_truth)} vehicles "
          f"({sum(r['wrong'] for r in ground_truth)} wrong-way)")
    print(f"Predicted violations: {len(violations)}")
    print()
    print("              Predicted YES   Predicted NO")
    print(f"Actual YES    {metrics['TP']:>12}   {metrics['FN']:>12}")
    print(f"Actual NO     {metrics['FP']:>12}   {metrics['TN']:>12}")
    print()
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:>10}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
