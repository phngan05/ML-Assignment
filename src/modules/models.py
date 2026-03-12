"""
models.py – Xây dựng và huấn luyện mô hình phân loại đa nhãn.

Pipeline:
    Đặc trưng (TF-IDF / BERT)  →  OneVsRestClassifier  →  Dự đoán nhãn

Mô hình hỗ trợ:
    - LogisticRegression
    - NaiveBayes  (ComplementNB cho TF-IDF  |  GaussianNB cho BERT)
    - LinearSVC
"""

import os
import joblib

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.svm import LinearSVC

# Thư mục mặc định lưu mô hình
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def build_models(
    feature_type: str = "tfidf",
    lr_C: float = 1.0,
    lr_max_iter: int = 1000,
    nb_alpha: float = 0.1,
    svc_C: float = 1.0,
    svc_max_iter: int = 2000,
) -> dict:
    """
    Tạo dict các mô hình OneVsRestClassifier với tham số cấu hình linh hoạt.

    Tham số:
        feature_type : "tfidf" → dùng ComplementNB  (sparse, non-negative)
                       "bert"  → dùng GaussianNB    (dense, có thể âm)
        lr_C         : Hệ số nghịch đảo chính quy hóa LogisticRegression
        lr_max_iter  : Số vòng lặp tối đa  LogisticRegression
        nb_alpha     : Tham số làm trơn cho ComplementNB (chỉ dùng khi tfidf)
        svc_C        : Hệ số nghịch đảo chính quy hóa LinearSVC
        svc_max_iter : Số vòng lặp tối đa  LinearSVC
    """
    if feature_type == "tfidf":
        # TF-IDF → sparse matrix, giá trị không âm → ComplementNB
        naive_bayes = ComplementNB(alpha=nb_alpha)
    else:
        # BERT   → dense matrix, giá trị có thể âm → GaussianNB
        naive_bayes = GaussianNB()

    return {
        "LogisticRegression": OneVsRestClassifier(
            LogisticRegression(C=lr_C, max_iter=lr_max_iter, solver="lbfgs")
        ),
        "NaiveBayes": OneVsRestClassifier(naive_bayes),
        "LinearSVC": OneVsRestClassifier(
            LinearSVC(C=svc_C, max_iter=svc_max_iter)
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """
    Huấn luyện tất cả mô hình trong dict.

    Trả về:
        dict {model_name: trained_model}
    """
    trained = {}
    for name, model in models.items():
        print(f"  Đang huấn luyện {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("✓")
        trained[name] = model
    return trained


def save_models(trained_models: dict, save_dir: str = MODELS_DIR, prefix: str = "") -> None:
    """Lưu các mô hình đã huấn luyện ra file .joblib."""
    os.makedirs(save_dir, exist_ok=True)
    for name, model in trained_models.items():
        path = os.path.join(save_dir, f"{prefix}{name}.joblib")
        joblib.dump(model, path)
        print(f"  Đã lưu: {path}")


def load_model(filename: str, load_dir: str = MODELS_DIR):
    """Tải mô hình từ file .joblib."""
    return joblib.load(os.path.join(load_dir, filename))
