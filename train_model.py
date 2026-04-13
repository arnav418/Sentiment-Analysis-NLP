import pandas as pd
import pickle
import string
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Load data
df = pd.read_csv('data/reviews.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove numbers & symbols
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['review'] = df['review'].apply(preprocess)

# TF-IDF (IMPROVED)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),  # 👈 VERY IMPORTANT
    stop_words='english'
)

X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Model (IMPROVED)
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # 👈 IMPORTANT
)

model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("Improved model trained and saved!")