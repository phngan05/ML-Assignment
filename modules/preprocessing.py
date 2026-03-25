import spacy

def lemma_data(text):
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    processed_texts = []
    for doc in nlp.pipe(df.astype(str), batch_size=500, n_process=-1):
        tokens = [t.lemma_ for t in doc if t.pos_ in ['NOUN', 'ADJ', 'VERB']]
        processed_texts.append(" ".join(tokens))
    return processed_texts
