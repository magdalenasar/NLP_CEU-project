#!/usr/bin/env python3
# Install dependencies
import subprocess
import sys

def install_and_download_models():
    try:
        # Install scikit-learn and spacy
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "spacy"])
        # Download SpaCy model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError as e:
        print(f"Error during installation/download: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError: # Handles if pip or python -m spacy is not found
        print("Error: pip or spacy command not found. Please ensure Python and pip are installed correctly and spacy is available.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # This check ensures that install_and_download_models() is called only when the script is executed directly,
    # not when it's imported as a module.
    install_and_download_models()

# Original script content follows...
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

"""
This script performs topic modeling on the 'data' file.
"""

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # Keeping sentence segmentation

# Load Data
def load_data(filepath):
    """Loads data from the specified filepath."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

document_text = load_data("data")

# Preprocess Text
def preprocess_sentence(sentence_text):
    """Preprocessing text for a single sentence: lowercasing, removing punctuation, tokenizing, lemmatizing, and removing stop words."""
    # We expect sentence_text to be a string (a single sentence)
    text = sentence_text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Since sentence_text is already a sentence, we process it directly
    # We need to re-process with nlp for lemmatization if we are not passing a Doc object
    doc = nlp(text) # Process the already lowercased and punctuation-removed sentence string
    lemmatized_tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and token.is_alpha]
    return " ".join(lemmatized_tokens)

# Process document into sentences and then preprocess each sentence
doc_for_sentencizer = nlp(document_text) # Use nlp once for sentence splitting
corpus = []
for sent in doc_for_sentencizer.sents:
    processed_sentence = preprocess_sentence(sent.text)
    if processed_sentence: # Avoid adding empty strings if a sentence becomes empty after processing
        corpus.append(processed_sentence)

# Vectorize Text
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# Apply Topic Model
n_topics = 7
nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=500, l1_ratio=0.0, solver='cd') # Added l1_ratio and solver for potential Frobenius norm issues
nmf_model.fit(tfidf_matrix)

# Save Topics
n_top_words = 12
with open("topics_output.txt", "w") as f:
    for topic_idx, topic_weights in enumerate(nmf_model.components_):
        top_word_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        f.write(f"Topic #{topic_idx + 1}:\n")
        f.write(", ".join(top_words) + "\n\n")

print("Topics saved to topics_output.txt")
