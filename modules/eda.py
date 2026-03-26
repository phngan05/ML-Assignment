import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
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

def visualize_corr_labels(df, column_name):
    """
    Correlation matrix between labels
    Args:
        df (pandas.DataFrame): DataFrame containing the labels
        column_name (str): Name of the column containing the labels
    """
    
    corr_matrix = df[column_name].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Matrix Between Labels")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_labels_per_article(df, column_name):
    """
    Visualize the distribution of labels per article.
    Args:
        df (pandas.DataFrame): DataFrame containing the labels
        column_name (str): Name of the column containing the labels
    """
    df['num_labels'] = df[column_name].sum(axis=1)
    df['num_labels'].value_counts().sort_index()

    df['num_labels'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Number of labels")
    plt.ylabel("Number of papers")
    plt.title("Distribution of labels per paper")
    plt.show()

def get_top_ngram(corpus, n=None, top_k=10):
    """
    corpus: df_clean['text_clean']
    n: 2 for bi-gram, 3 for tri-gram
    """
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return pd.DataFrame(words_freq[:top_k], columns=['N-gram', 'Frequency'])
  

def visualize_ngram(df, labels, n):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) 
    axes = axes.flatten() 

    for i, label in enumerate(labels):
        subset_df = df[df[label] == 1]
        
        top_ngrams = get_top_ngram(subset_df['lemma_text'], n=n, top_k=20)
        
        sns.barplot(x='Frequency', y='N-gram', data=top_ngrams, palette='viridis', ax=axes[i])
        
        axes[i].set_title(f'Top 20 {n}-grams: {label}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('frequency', fontsize=10)
        axes[i].set_ylabel('', fontsize=10)

    plt.tight_layout()
    plt.show()