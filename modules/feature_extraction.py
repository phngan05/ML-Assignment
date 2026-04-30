import torch
import numpy as np
import gc # Để giải phóng bộ nhớ
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def show_feature_extraction_top(vectorizer, method_name, X, doc_idx):
    feature_names = vectorizer.get_feature_names_out()
    vec = X[doc_idx].toarray().flatten()
    top = vec.argsort()[-5:][::-1]
    print(f"{method_name} - Top 5:")
    count = 1
    for idx in top:
        if vec[idx] > 0:
            print(f"  {count}. {feature_names[idx]:20s} → {vec[idx]:.0f}")
            count += 1



def extract_embeddings(text_list, model_id, max_length, device, batch_size=128):
    # Tải tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    all_embeddings = []
    # Chia nhỏ dữ liệu để tránh tràn RAM/GPU
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            # Lấy vector của [CLS] token (đại diện cho cả câu)
            # outputs.last_hidden_state có dạng [batch_size, seq_len, 768]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    # Giải phóng GPU/RAM sau khi xong mỗi model
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(all_embeddings)
