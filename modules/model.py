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
