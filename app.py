import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# Load model
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# 🔥 Custom CSS (BLACK THEME + ALIGNMENT)
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1 {
            text-align: center;
            font-size: 10rem;
        }
        .subtext {
            text-align: center;
            font-size: 1.1rem;
            color: #bbbbbb;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 45px;
            font-size: 16px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Sentiment Analysis with NLP</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Analyze text sentiment in real-time</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
text = st.text_area(
    "",
    height=150,
    placeholder="Type your text here..."
)

# Buttons (SIDE BY SIDE + SYMMETRY)
col1, col2 = st.columns(2)

with col1:
    analyze = st.button("Analyze")

with col2:
    clear = st.button("Clear")

# Clear functionality
if clear:
    st.session_state.clear()
    st.rerun()

# Prediction
if analyze:
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = preprocess(text)
        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        st.markdown("---")

        if prediction == "positive":
            st.success("Sentiment: Positive 😊")
            st.write(f"Confidence Score: {max(prob):.2f}")
        else:
            st.error("Sentiment: Negative 😠")
            st.write(f"Confidence Score: {max(prob):.2f}")

# Examples (SYMMETRIC)
st.markdown("---")
st.subheader("Try Examples")

col3, col4 = st.columns(2)

with col3:
    if st.button("Positive Example"):
        st.info("This product is amazing and works perfectly!")

with col4:
    if st.button("Negative Example"):
        st.info("This is the worst experience I’ve ever had.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built by Arnav Priyadarshi</p>", unsafe_allow_html=True)