import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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



def visualize_length_by_label(summary, df):
    """
    Visualize average text lengths by label using bar charts.
    
    Args:
        summary (list): List of dictionaries from calculate_length_by_label
        df (pandas.DataFrame): Original DataFrame for overall averages
    """
    import numpy as np
    
    labels_sorted = [s["label"] for s in sorted(summary, key=lambda x: x["abs_words"], reverse=True)]
    abs_words = [next(s["abs_words"] for s in summary if s["label"] == l) for l in labels_sorted]
    abs_chars = [next(s["abs_chars"] for s in summary if s["label"] == l) for l in labels_sorted]
    title_words = [next(s["title_words"] for s in summary if s["label"] == l) for l in labels_sorted]
    title_chars = [next(s["title_chars"] for s in summary if s["label"] == l) for l in labels_sorted]

    x = np.arange(len(labels_sorted))
    short_labels = [l.replace("Quantitative ", "Quant.\n") for l in labels_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Average length comparation by field", fontsize=14, fontweight="bold")

    datasets = [
        (axes[0, 0], abs_words, "ABSTRACT - Average number of words", "coral"),
        (axes[0, 1], abs_chars, "ABSTRACT - Average number of characters", "tomato"),
        (axes[1, 0], title_words, "TITLE - Average number of words", "steelblue"),
        (axes[1, 1], title_chars, "TITLE - Average number of characters", "royalblue"),
    ]

    for ax, values, title, color in datasets:
        bars = ax.bar(x, values, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel("Average value")
        
        # Overall average line
        if "ABSTRACT" in title and "words" in title:
            overall_val = df["abstract_word_len"].mean()
        elif "ABSTRACT" in title:
            overall_val = df["abstract_char_len"].mean()
        elif "TITLE" in title and "words" in title:
            overall_val = df["title_word_len"].mean()
        else:
            overall_val = df["title_char_len"].mean()

        ax.axhline(overall_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"Overall Average: {overall_val:.1f}")
        ax.legend(fontsize=8)
        for bar_rect, val in zip(bars, values):
            ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + overall_val * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("=" * 80)
    print("AVERAGE LENGTH COMPARATION BY FIELD")
    print("=" * 80)
    print(f"{'Field':<25} {'Number of articles':>7} "
          f"{'Title (words)':>11} {'Title (characters)':>14} "
          f"{'Abstract (words)':>14} {'Abstract (characters)':>17}")
    print("-" * 80)
    
    for s in summary:
        print(f"{s['label']:<25} {s['n']:>7,} "
              f"{s['title_words']:>11.1f} {s['title_chars']:>14.1f} "
              f"{s['abs_words']:>14.1f} {s['abs_chars']:>17.1f}")
    
    # Print observations
    print("\n--- Comment ---")
    max_abs = max(summary, key=lambda x: x["abs_words"])
    min_abs = min(summary, key=lambda x: x["abs_words"])
    max_title = max(summary, key=lambda x: x["title_words"])
    min_title = min(summary, key=lambda x: x["title_words"])
    print(f"Longest Abstract (words) : {max_abs['label']} ({max_abs['abs_words']:.1f} words/article)")
    print(f"Shortest Abstract (words): {min_abs['label']} ({min_abs['abs_words']:.1f} words/article)")
    print(f"Longest Title  (words)   : {max_title['label']} ({max_title['title_words']:.1f} words/article)")
    print(f"Shortest Title (words)   : {min_title['label']} ({min_title['title_words']:.1f} words/article)")
   
def visualize_frequency(df, column):
    all_words = " ".join(df[column]).split()
    top20_all = Counter(all_words).most_common(20)
    words_all, counts_all = zip(*top20_all)

    plt.figure(figsize=(12, 5))
    plt.bar(words_all, counts_all, color="steelblue", edgecolor="white")
    plt.title("Top 20 most frequent words - General", fontsize=13, fontweight="bold")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(f"{'Word':<20} {'Frequency':>10}")
    print("-" * 32)
    for word, cnt in top20_all:
        print(f"{word:<20} {cnt:>10,}")
 
def visualize_frequency_by_label(df, label_cols):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Top 20 most frequent words by label", fontsize=14, fontweight="bold")

    colors = ["steelblue", "mediumpurple", "seagreen", "coral", "deeppink", "goldenrod"]

    for ax, label, color in zip(axes.flatten(), label_cols, colors):
        words_label = " ".join(df[df[label] == 1]["text_clean"]).split()
        top20 = Counter(words_label).most_common(20)
        if not top20:
            continue
        words, counts = zip(*top20)
        ax.barh(list(words)[::-1], list(counts)[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Frequency")

    plt.tight_layout()
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
        
        sns.barplot(x='Frequency', y='N-gram', data=top_ngrams, palette='viridis', hue="N-gram", legend=False, ax=axes[i])
        
        axes[i].set_title(f'Top 20 {n}-grams: {label}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('frequency', fontsize=10)
        axes[i].set_ylabel('', fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_wordcloud(df, label_columns, text_column='text_clean', palette=None):
    """
    Visualize WordCloud for each label in a multi-label classification dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        label_columns (list): List of label column names
        text_column (str): Name of the column containing cleaned text
        palette (list): List of colormaps for each label (optional)
    """
    
    if palette is None:
        palette = ["Blues_r", "Purples_r", "Greens_r", "Oranges_r", "PuRd_r", "YlOrRd_r"]
    
    # Ensure we have enough colors for all labels
    while len(palette) < len(label_columns):
        palette = palette + palette
    
    def make_wordcloud(text, title, ax, colormap="viridis"):
        wc = WordCloud(
            width=600, height=350,
            background_color="white",
            max_words=100,
            colormap=colormap,
            collocations=False,
        ).generate(text if text.strip() else "no data")
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("WordCloud by label", fontsize=14, fontweight="bold", y=1.01)
    
    for ax, label, cmap in zip(axes.flatten(), label_columns, palette[:len(label_columns)]):
        text = " ".join(df[df[label] == 1][text_column].dropna())
        make_wordcloud(text, label, ax, cmap)
    
    plt.tight_layout()
    plt.show()
    
# ================ FEATURE EXTRACTION =================

def visualize_tfidf_top(top_scores, top_words, X_tfidf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(x=top_scores, y=top_words, palette='viridis', ax=axes[0])
    axes[0].set_xlabel('Avg TF-IDF Score')
    axes[0].set_title('Top 15 Terms by TF-IDF')
    axes[0].grid(axis='x', alpha=0.3)

    # Distribution
    non_zero = X_tfidf.data
    axes[1].hist(non_zero, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('TF-IDF Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('TF-IDF Value Distribution')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"TF-IDF stats: min={non_zero.min():.4f}, max={non_zero.max():.4f}, mean={non_zero.mean():.4f}")
    
    
def visualize_models_comparation(results):
    # Bar chart
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, results['Micro-F1'], width, label='Micro-F1', color='steelblue')
    ax.bar(x + width/2, results['Macro-F1'], width, label='Macro-F1', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    ax.set_title('LinearSVC vs Random Forest\n(Deep Learning Feature Extraction)')
    ax.legend()
    for bar in ax.patches:
        ax.annotate(f'{bar.get_height():.3f}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()