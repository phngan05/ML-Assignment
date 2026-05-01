import time
from sklearn.metrics import f1_score, classification_report
import numpy as np

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
    
def find_best_threshold(probs, threshold_range, y_test):
    thresholds = []
    for i in range(6):
        best_t, best_f1 = 0.5, 0
        for t in threshold_range:
            pred = (probs[:, i] > t).astype(int)
            f = f1_score(y_test[:, i], pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
        
    return thresholds
