"""
SMD Metrics Calculator
Reads SMD .txt files directly and computes Precision, Recall, F1, Accuracy
Outputs results to CSV
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score)
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG — adjust path if needed
# ─────────────────────────────────────────────
SMD_PATH = r"C:\Users\Admin\Desktop\PAPERS\ma'am folder\AFMF-main\AFMF-main\data\SMD"
OUTPUT_DIR = r"C:\Users\Admin\Desktop\PAPERS\ma'am folder\AFMF-main\AFMF-main\benchmark_results"
THRESHOLD_PERCENTILE = 95   # anomaly score threshold (top 5% flagged as anomaly)


def load_smd_machine(smd_path, machine_name):
    """Load test data + labels for one machine."""
    test_file  = os.path.join(smd_path, "test",       machine_name + ".txt")
    label_file = os.path.join(smd_path, "test_label", machine_name + ".txt")

    test_data = np.loadtxt(test_file,  delimiter=",")
    labels    = np.loadtxt(label_file, delimiter=",").astype(int)
    return test_data, labels


def compute_anomaly_scores(test_data):
    """
    Simple reconstruction-error anomaly score using row-wise z-score distance.
    Replace this function with your model's actual inference if needed.
    """
    mean = test_data.mean(axis=0)
    std  = test_data.std(axis=0) + 1e-8
    z    = np.abs((test_data - mean) / std)
    scores = z.mean(axis=1)          # mean z-score across all features per timestep
    return scores


def scores_to_predictions(scores, percentile=95):
    """Convert continuous anomaly scores to binary predictions."""
    threshold = np.percentile(scores, percentile)
    return (scores >= threshold).astype(int)


def evaluate_machine(smd_path, machine_name):
    """Run evaluation for a single machine and return metrics dict."""
    try:
        test_data, labels = load_smd_machine(smd_path, machine_name)
    except Exception as e:
        print(f"  [SKIP] {machine_name}: {e}")
        return None

    scores      = compute_anomaly_scores(test_data)
    predictions = scores_to_predictions(scores, THRESHOLD_PERCENTILE)

    # Guard: if labels are all one class, AUC is undefined
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    metrics = {
        "machine":   machine_name,
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall":    recall_score(labels, predictions, zero_division=0),
        "f1_score":  f1_score(labels, predictions, zero_division=0),
        "accuracy":  accuracy_score(labels, predictions),
        "auc_roc":   auc,
        "n_samples": len(labels),
        "n_anomalies_true": int(labels.sum()),
        "n_anomalies_pred": int(predictions.sum()),
    }
    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover all machine names from test_label folder
    label_dir = os.path.join(SMD_PATH, "test_label")
    machines  = sorted([
        f.replace(".txt", "")
        for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ])

    if not machines:
        print(f"[ERROR] No .txt files found in {label_dir}")
        return

    print(f"Found {len(machines)} machines. Running evaluation...\n")

    results = []
    for machine in machines:
        print(f"  Processing: {machine}")
        m = evaluate_machine(SMD_PATH, machine)
        if m:
            results.append(m)
            print(f"    F1={m['f1_score']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  ACC={m['accuracy']:.4f}")

    if not results:
        print("[ERROR] No results computed.")
        return

    df = pd.DataFrame(results)

    # Aggregate row
    agg = {
        "machine":            "AVERAGE",
        "precision":          df["precision"].mean(),
        "recall":             df["recall"].mean(),
        "f1_score":           df["f1_score"].mean(),
        "accuracy":           df["accuracy"].mean(),
        "auc_roc":            df["auc_roc"].mean(),
        "n_samples":          df["n_samples"].sum(),
        "n_anomalies_true":   df["n_anomalies_true"].sum(),
        "n_anomalies_pred":   df["n_anomalies_pred"].sum(),
    }
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    # Save CSV
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(OUTPUT_DIR, f"smd_metrics_{timestamp}.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\n" + "="*55)
    print("OVERALL AVERAGE METRICS")
    print("="*55)
    print(f"  Precision : {agg['precision']:.4f}")
    print(f"  Recall    : {agg['recall']:.4f}")
    print(f"  F1-Score  : {agg['f1_score']:.4f}")
    print(f"  Accuracy  : {agg['accuracy']:.4f}")
    print(f"  AUC-ROC   : {agg['auc_roc']:.4f}")
    print("="*55)
    print(f"\n✓ CSV saved to:\n  {csv_path}")


if __name__ == "__main__":
    main()
