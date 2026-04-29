import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize


def calculate_text_lengths(df, column_name):
    """
    Calculates the necessary lengths in a specified text column.
    """
    # Calculate the number of characters (including spaces)
    df[f'{column_name}_chars'] = df[column_name].str.len()
    
    # Calculate the number of words (split by spaces)
    df[f'{column_name}_words'] = df[column_name].apply(lambda x: len(str(x).split()))
    
    # Calculate the mean word length
    df[f'{column_name}_mean_word_length'] = df[f'{column_name}_chars'] / df[f'{column_name}_words']
    
    # Calculate mean sentence length
    df[f'{column_name}_mean_sentence_length'] = df[column_name].map(lambda x: np.mean([len(sent) for sent in sent_tokenize(str(x))]))
    
    return df

def calculate_length_by_label(df, label_columns):
    """
    Calculate average text lengths (words and characters) for each label.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the text and labels
        label_columns (list): List of label column names
    
    Returns:
        list: List of dictionaries containing summary statistics for each label
    """
    # Ensure length columns exist
    if "abstract_word_len" not in df.columns:
        df["title_char_len"] = df["TITLE"].str.len()
        df["abstract_char_len"] = df["ABSTRACT"].str.len()
        df["title_word_len"] = df["TITLE"].str.split().str.len()
        df["abstract_word_len"] = df["ABSTRACT"].str.split().str.len()
    
    summary = []
    for lbl in label_columns:
        sub = df[df[lbl] == 1]
        summary.append({
            "label": lbl,
            "n": len(sub),
            "title_words": sub["title_word_len"].mean(),
            "title_chars": sub["title_char_len"].mean(),
            "abs_words": sub["abstract_word_len"].mean(),
            "abs_chars": sub["abstract_char_len"].mean(),
        })
    
    return summary
