#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#Loading dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[['label', 'message']]
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

#Preprocessing
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Text Vectorization(TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Naive Bayes Classifier Algorithm
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy for Naive Bayes: {accuracy_nb * 100:.2f}%")
print("\nClassification Report for Naive Bayes:\n", classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix for Naive Bayes:\n", confusion_matrix(y_test, y_pred_nb))

#Logistic Regression Classifier Algorithm
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy for Logistic Regression: {accuracy_lr * 100:.2f}%")
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix for Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))

#Testing with custom message
custom_message = input("\nEnter an SMS to classify as 'ham' or 'spam': ")
custom_message_tfidf = tfidf.transform([custom_message])

#Naive Bayes Prediction
custom_prediction_nb = nb_model.predict(custom_message_tfidf)
if custom_prediction_nb[0] == 1:
    print("Naive Bayes Classifier says: Spam")
else:
    print("Naive Bayes Classifier says: Ham")

#Logistic Regression Prediction
custom_prediction_lr = lr_model.predict(custom_message_tfidf)
if custom_prediction_lr[0] == 1:
    print("Logistic Regression Classifier says: Spam")
else:
    print("Logistic Regression Classifier says: Ham")
