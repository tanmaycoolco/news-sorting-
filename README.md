# news-sorting-
news article sorting using matpoltlib python
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Initialize a Support Vector Machine (SVM) classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Training data (text samples) and their corresponding labels
X_train = [
    "Stock markets reached a new high today...",
    "The football match ended in a tie...",
    "New technology breakthrough announced...",
]
y_train = [0, 1, 2]  # Corresponding labels for the training data

# Initialize a TF-IDF vectorizer, which will convert text data into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the training data into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train)

# New text data for prediction
news_text = ["Rohit Sharma’s side rested a couple of players and possibly experiment with their playing 11 as suggested by bowling coach Paras Mhambrey on the eve of the match."]

# Transform the new text data into TF-IDF features using the same vectorizer
news_vector = vectorizer.transform(news_text)

# Train the SVM classifier using the TF-IDF transformed training data and labels
svm_classifier.fit(X_train_tfidf, y_train)

# Predict the category for the new text data
predicted_category = svm_classifier.predict(news_vector)[0]

# Labels for the categories
category_labels = ["Finance", "Sports", "Technology"]

# Create a bar chart to visualize the confidence scores for each category
plt.bar(category_labels, svm_classifier.decision_function(news_vector)[0])
plt.xlabel("Categories")
plt.ylabel("Confidence Score")
plt.title(f"Predicted Category: {category_labels[predicted_category]}")
plt.xticks(rotation=45)
plt.show()

