"""
feature_extraction.py – Trích xuất đặc trưng văn bản cho bài toán phân loại đa nhãn.

Hai phương pháp:
  1. TF-IDF  – truyền thống, nhanh, phù hợp baseline
  2. BERT    – hiện đại, dùng SentenceTransformer 'all-MiniLM-L6-v2'

Đầu ra được lưu dưới dạng .npy vào thư mục src/features/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz


# ── Đường dẫn mặc định ────────────────────────────────────────
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  1. TF-IDF
# ══════════════════════════════════════════════════════════════

def build_tfidf(
    train_texts: pd.Series,
    val_texts: pd.Series,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
    save_dir: str = None,
) -> tuple:
    """
    Fit TfidfVectorizer trên train, transform cả train và val.

    Trả về:
        (vectorizer, X_train_tfidf, X_val_tfidf)

    Nếu save_dir được cung cấp, lưu:
        tfidf_train.npz, tfidf_val.npz
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    X_val   = vectorizer.transform(val_texts.fillna(""))

    if save_dir:
        _ensure_dir(save_dir)
        save_npz(os.path.join(save_dir, "tfidf_train.npz"), X_train)
        save_npz(os.path.join(save_dir, "tfidf_val.npz"),   X_val)
        print(f"[TF-IDF] Đã lưu → {save_dir}/tfidf_train.npz  &  tfidf_val.npz")

    return vectorizer, X_train, X_val


def load_tfidf(load_dir: str = None) -> tuple:
    """Tải lại ma trận TF-IDF đã lưu."""
    d = load_dir or FEATURES_DIR
    X_train = load_npz(os.path.join(d, "tfidf_train.npz"))
    X_val   = load_npz(os.path.join(d, "tfidf_val.npz"))
    return X_train, X_val


# ══════════════════════════════════════════════════════════════
#  2. BERT  (SentenceTransformer)
# ══════════════════════════════════════════════════════════════

def build_bert_embeddings(
    train_texts: pd.Series,
    val_texts: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    save_dir: str = None,
) -> tuple:
    """
    Encode văn bản bằng SentenceTransformer.

    Trả về:
        (X_train_bert, X_val_bert)  – numpy arrays shape (n, 384)

    Nếu save_dir được cung cấp, lưu:
        bert_train.npy, bert_val.npy
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Cần cài đặt sentence-transformers:\n"
            "  pip install sentence-transformers"
        )

    print(f"[BERT] Đang load model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"[BERT] Encoding train ({len(train_texts):,} mẫu)...")
    X_train = model.encode(
        train_texts.fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"[BERT] Encoding val ({len(val_texts):,} mẫu)...")
    X_val = model.encode(
        val_texts.fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if save_dir:
        _ensure_dir(save_dir)
        np.save(os.path.join(save_dir, "bert_train.npy"), X_train)
        np.save(os.path.join(save_dir, "bert_val.npy"),   X_val)
        print(f"[BERT] Đã lưu → {save_dir}/bert_train.npy  &  bert_val.npy")

    return X_train, X_val


def load_bert(load_dir: str = None) -> tuple:
    """Tải lại BERT embeddings đã lưu."""
    d = load_dir or FEATURES_DIR
    X_train = np.load(os.path.join(d, "bert_train.npy"))
    X_val   = np.load(os.path.join(d, "bert_val.npy"))
    return X_train, X_val
