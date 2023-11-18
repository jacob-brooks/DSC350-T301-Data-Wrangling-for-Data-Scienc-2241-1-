# %%
pip install pandas scikit-learn nltk

# %%
import nltk

# Download the punkt resource
nltk.download('punkt')

# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download stopwords and punkt resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
file_path = 'C:/Users/JacobBrooks/Downloads/twitter_sample.csv'
df = pd.read_csv(file_path)

# Clean the "Tweet Content" column by removing non-text data and stop words
def clean_text(text):
    # Remove non-text data
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Apply the cleaning function to the "Tweet Content" column
df['cleaned_tweet'] = df['Tweet Content'].apply(clean_text)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

# Filter only tweets (not re-tweets)
df = df[~df['Tweet Content'].str.startswith('RT')]

# Build Bag of Words (BOW) representation
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(df['cleaned_tweet'])

# Build TF-IDF representation
vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(df['cleaned_tweet'])

# Print BOW and TF-IDF results
print("Bag of Words Matrix:")
print(bow_matrix.toarray())
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Find similar documents using Cosine Similarity
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Find documents that are similar to each other
similar_docs_indices = [(i, j) for i in range(cosine_sim_matrix.shape[0]) for j in range(i+1, cosine_sim_matrix.shape[1]) if cosine_sim_matrix[i, j] > 0.8]

# Print results
for i, j in similar_docs_indices:
    print(f"\nSimilarity between Document {i+1} and Document {j+1}: {cosine_sim_matrix[i, j]}")
    print(f"Document {i+1}:\n{df['cleaned_tweet'].iloc[i]}")
    print(f"Document {j+1}:\n{df['cleaned_tweet'].iloc[j]}")



