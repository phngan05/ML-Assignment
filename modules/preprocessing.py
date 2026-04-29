import spacy
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    
def remove_noise(text: str) -> str:
    # Xóa ký hiệu LaTeX: $...$  $$...$$  \command{...}  \command
    text = re.sub(r'\$\$?.*?\$\$?', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    # Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Xóa số
    text = re.sub(r'\d+', ' ', text)

    # Xóa ký tự đặc biệt và dấu câu (chỉ giữ chữ cái và khoảng trắng)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    STOPWORDS = set(stopwords.words("english"))
    
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def lemma_data(df, column):
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    processed_texts = []
    for doc in nlp.pipe(df[column].astype(str), batch_size=500, n_process=-1):
        tokens = [t.lemma_ for t in doc if t.pos_ in ['NOUN', 'ADJ', 'VERB']]
        processed_texts.append(" ".join(tokens))
    return processed_texts
