import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

# Load the dataset
dataset = pd.read_csv('./Test.csv')  # Ensure your CSV path is correct

# Preprocessing text data
nltk.download('stopwords')
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')  # Keep 'not' for negation

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

# Convert text into numerical features using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=1500)
# X = tfidf_vectorizer.fit_transform(corpus).toarray()
# y = dataset.iloc[:, -1].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# Define hyperparameters for Grid Search with additional parameters
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3]  # For polynomial kernel
}

classifier = SVC(random_state=21)
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = best_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Confusion Matrix:\n", cm)
print("Test Set Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
