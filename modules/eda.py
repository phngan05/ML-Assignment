import numpy as np
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns


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

def visualize_text_length_distribution(df, column_name):
    """
    Visualize the distribution of text lengths using boxplot and histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y = df[column_name], color="red", ax=axes[0])
    sns.histplot(df[column_name], color="skyblue", kde=True, ax=axes[1])
    fig.suptitle(f'Distribution of {column_name}', fontsize=16)
    plt.show()