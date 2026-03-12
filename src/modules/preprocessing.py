"""
preprocessing.py – Tiền xử lý văn bản cho bài toán phân loại đa nhãn bài báo khoa học.

Chức năng:
- Ghép TITLE + ABSTRACT thành một cột văn bản tổng hợp
- Làm sạch văn bản: chữ thường, loại ký tự đặc biệt/LaTeX, loại stopwords
- Chia dữ liệu train / val theo tỷ lệ định sẵn
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

# Danh sách nhãn cố định
LABEL_COLS = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

# Stopwords tiếng Anh (không cần NLTK để tránh phụ thuộc)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "that", "this", "these", "those", "it", "its",
    "we", "our", "they", "their", "he", "she", "his", "her", "i", "my",
    "you", "your", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "also", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "few", "while", "which", "who", "whom", "s", "t", "d", "ll",
    "re", "ve", "m", "about", "up", "down", "what",
}


def clean_text(
    text: str,
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    remove_latex: bool = True,
    min_word_len: int = 3,
) -> str:
    """
    Làm sạch một chuỗi văn bản với các tham số cấu hình linh hoạt.

    Tham số:
        remove_stopwords : Loại bỏ stopwords tiếng Anh (mặc định: True)
        remove_numbers   : Loại bỏ chữ số (mặc định: True)
        remove_latex     : Loại bỏ ký hiệu LaTeX $...$ và \\cmd (mặc định: True)
        min_word_len     : Độ dài từ tối thiểu được giữ lại (mặc định: 3)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    if remove_latex:
        # Xóa LaTeX inline/block: $...$, $$...$$
        text = re.sub(r"\$\$?.*?\$\$?", " ", text, flags=re.DOTALL)
        # Xóa lệnh LaTeX: \command{...} hoặc \command
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Xóa URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # Chỉ giữ lại chữ cái và khoảng trắng
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tách từ, lọc theo độ dài tối thiểu và stopwords
    tokens = [
        w for w in text.split()
        if len(w) >= min_word_len
        and (not remove_stopwords or w not in STOPWORDS)
    ]

    return " ".join(tokens)


def combine_and_clean(
    df: pd.DataFrame,
    title_col: str = "TITLE",
    abstract_col: str = "ABSTRACT",
    sep: str = " ",
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    remove_latex: bool = True,
    min_word_len: int = 3,
) -> pd.DataFrame:
    """
    Ghép TITLE + ABSTRACT thành cột `text`, sau đó làm sạch thành cột `text_clean`.

    Tham số cấu hình pipeline:
        remove_stopwords : Loại bỏ stopwords (mặc định: True)
        remove_numbers   : Loại bỏ số (mặc định: True)
        remove_latex     : Loại bỏ ký hiệu LaTeX (mặc định: True)
        min_word_len     : Độ dài từ tối thiểu giữ lại (mặc định: 3)
    """
    df = df.copy()
    df["text"] = df[title_col].fillna("") + sep + df[abstract_col].fillna("")
    df["text_clean"] = df["text"].apply(
        lambda t: clean_text(
            t,
            remove_stopwords=remove_stopwords,
            remove_numbers=remove_numbers,
            remove_latex=remove_latex,
            min_word_len=min_word_len,
        )
    )
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia DataFrame thành (train, val) theo tỷ lệ (1-test_size) / test_size.
    Dùng stratify trên cột num_labels để giữ phân phối số nhãn tương đối đồng đều.
    """
    stratify_col = df[LABEL_COLS].sum(axis=1).clip(upper=2)  # nhóm 0,1,2+
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
