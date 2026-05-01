import time
from sklearn.metrics import f1_score, classification_report
import numpy as np

def tokenize_and_encode(examples,tokenizer):
    texts = [t + " [SEP] " + a for t, a in zip(examples['TITLE'], examples['ABSTRACT'])]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=256)

    # Nhãn phải ở dạng float cho bài toán Multi-label
    labels = []
    cols = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    for i in range(len(examples['TITLE'])):
        label_row = [float(examples[col][i]) for col in cols]
        labels.append(label_row)

    tokenized["labels"] = labels
    return tokenized


def train_and_predict(model, dataset):
    X_train, y_train, X_test, y_test = dataset
    t0 = time.time()
    
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test)

    return model, y_pred

def evaluate_model(y_pred, y_test, label_columns):
    print(f"Micro-F1 : {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"Macro-F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=label_columns, zero_division=0))
    
def find_best_threshold(probs, range, y_test):
    thresholds = []
    for i in range(6):
        best_t, best_f1 = 0.5, 0
        for t in range:
            pred = (probs[:, i] > t).astype(int)
            f = f1_score(y_test[:, i], pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
        
    return thresholds
