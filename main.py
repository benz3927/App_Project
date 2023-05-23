import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics

# Load the dataset
df = pd.read_csv("complaints.csv")

# Preprocessing the data
df = df[['text', 'airline_sentiment']]
df = df[df.airline_sentiment != 'neutral']

# Split the dataset into training and testing sets
X = df.text
y = df.airline_sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Support Vector Machine (SVM) model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
