# 🎯 Sentiment Analysis NLP

A machine learning-based web application that analyzes user text and predicts whether the sentiment is **Positive 😊** or **Negative 😡** using Natural Language Processing (NLP).

---

## 🚀 Live Demo

👉 (Add your Render link here after deployment)

---

## 📌 Features

- 🔍 Real-time sentiment prediction
- 🧠 NLP-based text preprocessing
- ⚡ Fast predictions using trained ML model
- 🎨 Clean and interactive Streamlit UI
- 📊 Logistic Regression model with TF-IDF vectorization

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK

---

## 📂 Project Structure

```
AI-Sentiment-Analyzer/
│── data/                 # Dataset
│── app.py               # Streamlit App
│── train_model.py       # Model Training Script
│── model.pkl            # Trained Model
│── tfidf.pkl            # TF-IDF Vectorizer
│── requirements.txt     # Dependencies
│── README.md            # Project Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/arnav418/AI-Sentiment-Analyzer.git
cd AI-Sentiment-Analyzer
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. User inputs text
2. Text is cleaned & preprocessed using NLP techniques
3. TF-IDF converts text → numerical vectors
4. Logistic Regression predicts sentiment
5. Result is displayed instantly

---

## 📊 Model Details

- Algorithm: Logistic Regression
- Vectorization: TF-IDF
- Type: Supervised Machine Learning
- Accuracy: ~85% (can vary based on dataset)

---

## 🎯 Future Improvements

- 🔥 Add Deep Learning (LSTM/BERT)
- 🌐 Deploy with database integration
- 📈 Add sentiment score visualization
- 🧾 Upload CSV for bulk predictions

---

## 👨‍💻 Author

**Arnav Priyadarshi**

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
