# ==========================================
# SMS Spam Detection System
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv(
    'spam.csv',
    encoding='latin-1'
)

# Keep useful columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

print("\nDataset Loaded Successfully!")

# ==========================================
# Convert Labels
# ham = 0
# spam = 1
# ==========================================

data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# ==========================================
# Features and Labels
# ==========================================

X = data['message']

y = data['label']

# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("\nTF-IDF Vectorization Completed!")

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Naive Bayes Model
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# Prediction
# ==========================================

pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:\n")

print(classification_report(y_test, pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Spam Detection Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()