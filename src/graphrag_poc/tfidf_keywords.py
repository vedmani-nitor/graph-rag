import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords

english_stopwords = list(set(stopwords.words('english')))

print(english_stopwords)

def extract_important_keywords(directory_path, threshold=0.05):
    """
    Extract important keywords from all txt files in a directory.
    
    Args:
        directory_path (str): Path to directory containing text files
        threshold (float): Minimum importance score threshold
    """
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        stop_words=english_stopwords,
        min_df=1  # Term must appear in at least 1 document
    )
    
    # Collect all texts
    texts = []
    filenames = []
    for file_path in Path(directory_path).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
            filenames.append(file_path.name)
    
    if not texts:
        print("No text files found in directory!")
        return
    
    # Get TF-IDF scores - creates a matrix where:
    # - Each row represents a document
    # - Each column represents a term
    # - Each cell contains the TF-IDF score of that term in that document
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate importance metrics for each term
    doc_freq = np.sum(tfidf_matrix.toarray() > 0, axis=0)  # Number of documents containing each term
    
    # max_scores finds the highest TF-IDF score each term got in any document
    # High max_score means the term was very important in at least one document
    # Example: If "machine learning" has max_score of 0.8, it was highly significant in at least one document
    max_scores = np.max(tfidf_matrix.toarray(), axis=0)
    
    # mean_scores finds the average TF-IDF score of each term across all documents
    # High mean_score means the term was consistently important across multiple documents
    # Example: If "data" has mean_score of 0.4, it was moderately important in many documents
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # Get keywords above threshold using max_score
    # This captures terms that were either:
    # 1. Very important in specific documents (high max_score)
    # 2. Moderately important across many documents (high mean_score)
    keywords = [
        (
            feature_names[i],          # The term or phrase
            max_scores[i],             # Highest importance in any document
            mean_scores[i],            # Average importance across documents
            doc_freq[i],              # Number of documents containing the term
            len(feature_names[i].split())  # Length of n-gram (1, 2, or 3 words)
        )
        for i in range(len(feature_names))
        if max_scores[i] >= threshold
    ]
    
    # Sort by max_score to prioritize terms that were highly important in at least one document
    keywords.sort(key=lambda x: x[1], reverse=True)
    
    # Save results to CSV with all metrics for analysis
    df = pd.DataFrame(
        keywords, 
        columns=['keyword', 'max_score', 'mean_score', 'document_frequency', 'gram_length']
    )
    df.to_csv("important_keywords.csv", index=False)
    print(f"Found {len(keywords)} important keywords. Results saved to 'important_keywords.csv'")
    
    # Print statistics about the types of keywords found
    print("\nKeyword statistics:")
    print(f"Unigrams: {len(df[df['gram_length'] == 1])}")
    print(f"Bigrams: {len(df[df['gram_length'] == 2])}")
    print(f"Trigrams: {len(df[df['gram_length'] == 3])}")
    print("\nTop 10 keywords by max score:")
    for _, row in df.head(10).iterrows():
        print(f"{row['keyword']}: {row['max_score']:.4f} (in {row['document_frequency']} docs)")

if __name__ == "__main__":
    extract_important_keywords("D:\projects\graphrag-poc\data\selection 1")