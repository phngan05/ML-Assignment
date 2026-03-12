"""
evaluation.py – Đánh giá và so sánh các mô hình phân loại đa nhãn.

Hàm chính:
    evaluate_model()       → Tính Micro-F1, Macro-F1, classification_report
    print_results_table()  → Bảng so sánh tất cả mô hình
    print_per_label_report() → Báo cáo chi tiết theo từng nhãn
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_model(model, X_val, y_val, label_names: list) -> dict:
    """
    Đánh giá một mô hình trên tập validation.

    Trả về dict:
        micro_f1 : float
        macro_f1 : float
        report   : dict (output_dict=True từ classification_report)
        y_pred   : np.ndarray
    """
    y_pred = model.predict(X_val)
    return {
        "micro_f1": f1_score(y_val, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "report": classification_report(
            y_val, y_pred,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
        "y_pred": y_pred,
    }


def print_results_table(results: dict) -> None:
    """
    In bảng so sánh Micro-F1 / Macro-F1 cho tất cả tổ hợp (feature × model).

    Key của results phải có dạng: "FEATURE_TYPE|MODEL_NAME"
    Ví dụ: "TF-IDF|LogisticRegression", "BERT|NaiveBayes"
    """
    rows = []
    for key, m in results.items():
        feat, model_name = key.split("|", 1)
        rows.append({
            "Feature":  feat,
            "Model":    model_name,
            "Micro-F1": m["micro_f1"],
            "Macro-F1": m["macro_f1"],
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("Micro-F1", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 66)
    print("  BẢNG SO SÁNH KẾT QUẢ (Validation Set)")
    print("=" * 66)
    print(f"  {'#':<4} {'Feature':<10} {'Model':<22} {'Micro-F1':>10} {'Macro-F1':>10}")
    print("  " + "-" * 60)
    for i, row in df.iterrows():
        marker = "  ← best" if i == 0 else ""
        print(
            f"  {i+1:<4} {row['Feature']:<10} {row['Model']:<22} "
            f"{row['Micro-F1']:>10.4f} {row['Macro-F1']:>10.4f}{marker}"
        )
    print()


def print_per_label_report(metrics: dict, label_names: list) -> None:
    """In classification report chi tiết theo từng nhãn."""
    report = metrics["report"]
    print(f"  {'Nhãn':<28} {'Precision':>10} {'Recall':>8} {'F1-score':>10} {'Support':>9}")
    print("  " + "-" * 68)
    for lbl in label_names:
        r = report.get(lbl, {})
        print(
            f"  {lbl:<28} {r.get('precision', 0):>10.3f} "
            f"{r.get('recall', 0):>8.3f} "
            f"{r.get('f1-score', 0):>10.3f} "
            f"{int(r.get('support', 0)):>9,}"
        )
    print("  " + "-" * 68)
    print(f"  {'micro avg':<28} {metrics['micro_f1']:>29.3f}   (Micro-F1)")
    print(f"  {'macro avg':<28} {metrics['macro_f1']:>29.3f}   (Macro-F1)")
