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