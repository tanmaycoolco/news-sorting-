import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

svm_classifier = SVC(kernel='linear')


X_train = [
    "Stock markets reached a new high today...",
    "The football match ended in a tie...",
    "New technology breakthrough announced...",
    
]

y_train = [0, 1, 2]


vectorizer = TfidfVectorizer(stop_words='english')


X_train_tfidf = vectorizer.fit_transform(X_train)


news_text = [ "Rohit Sharma’s side rested a couple of players and possibly experiment with their playing 11 as suggested by bowling coach Paras Mhambrey on the eve of the match. " ]


news_vector = vectorizer.transform(news_text)


svm_classifier.fit(X_train_tfidf, y_train)


predicted_category = svm_classifier.predict(news_vector)[0]


category_labels = ["Finance", "Sports", "Technology"]



plt.bar(category_labels, svm_classifier.decision_function(news_vector)[0])
plt.xlabel("Categories")
plt.ylabel("Confidence Score")
plt.title(f"Predicted Category: {category_labels[predicted_category]}")
plt.xticks(rotation=45)
plt.show()



