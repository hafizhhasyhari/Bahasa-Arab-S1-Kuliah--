# Install pustaka yang diperlukan
!pip install nltk scikit-learn arabic-reshaper bidi pandas

# Import pustaka
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import arabic_reshaper
from bidi.algorithm import get_display

# Download stopwords untuk bahasa Arab
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words("arabic"))

# Dataset (contoh sederhana, gunakan dataset yang lebih besar untuk akurasi lebih baik)
data = {
    "text": [
        "هذه قصة رائعة جدا",  # Positif
        "اللغة العربية لغة معقدة",  # Negatif
        "أنا سعيد بقراءة هذا النص",  # Positif
        "النص غير مفهوم وصعب",  # Negatif
    ],
    "sentiment": ["positive", "negative", "positive", "negative"]
}

# Membuat dataframe
df = pd.DataFrame(data)

# Preprocessing teks Arab
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in arabic_stopwords]
    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# Convert teks ke fitur menggunakan CountVectorizer
vectorizer = CountVectorizer(analyzer="word")
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# Membagi dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training menggunakan Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = model.predict(X_test)

# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Fungsi untuk prediksi sentimen input baru
def predict_sentiment(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)  # Untuk tampilan teks Arab yang benar
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return bidi_text, prediction[0]

# Contoh prediksi pada teks baru
new_text = "النص جميل ومليء بالعواطف"
display_text, sentiment = predict_sentiment(new_text)
print("\nInput Text:", display_text)
print("Predicted Sentiment:", sentiment)
