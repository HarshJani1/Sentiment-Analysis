import numpy as np
import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

# Load dataset
dataset = pd.read_csv('./Test.csv')

# Text preprocessing
nltk.download('stopwords')
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# SVM with GridSearch
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3]
}
svm = GridSearchCV(SVC(probability=True, random_state=21), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm.fit(X_train, y_train)
svm_best = svm.best_estimator_

# Random Forest with GridSearch
rf_param_grid = {
    'n_estimators': [100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
}
rf = GridSearchCV(RandomForestClassifier(random_state=21), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf.fit(X_train, y_train)
rf_best = rf.best_estimator_

# Naive Bayes with GridSearch
nb_param_grid = {
    'alpha': [0.1, 0.5, 1.0]
}
nb = GridSearchCV(MultinomialNB(), nb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
nb.fit(X_train, y_train)
nb_best = nb.best_estimator_

# Voting Classifier (soft voting)
voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf', rf_best),
        ('nb', nb_best)
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)

# K-Means clustering (unsupervised)
kmeans = KMeans(n_clusters=len(set(y)), random_state=21, n_init=10)
kmeans.fit(X_train)
kmeans_pred = kmeans.predict(X_test)

# Store evaluation results
results = {
    "Model": [],
    "Accuracy": [],
    "MAE": [],
    "MSE": [],
    "RMSE": []
}

# Evaluation function
def evaluate_model(name, model, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n--- {name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", acc)
    print("Precision:", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average='weighted', zero_division=0))
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["MAE"].append(mae)
    results["MSE"].append(mse)
    results["RMSE"].append(rmse)

# Predictions and Evaluation
evaluate_model("SVM", svm_best, y_test, svm_best.predict(X_test))
evaluate_model("Random Forest", rf_best, y_test, rf_best.predict(X_test))
evaluate_model("Naive Bayes", nb_best, y_test, nb_best.predict(X_test))
evaluate_model("Voting Classifier", voting_clf, y_test, voting_clf.predict(X_test))
evaluate_model("K-Means", kmeans, y_test, kmeans_pred)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot Accuracy Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='Set2')
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Error Metrics Comparison
plt.figure(figsize=(12, 6))
results_melted = results_df.melt(id_vars="Model", value_vars=["MAE", "MSE", "RMSE"], var_name="Error Metric", value_name="Value")
sns.barplot(x="Model", y="Value", hue="Error Metric", data=results_melted, palette='Set1')
plt.title("Model Error Metrics Comparison")
plt.ylabel("Error Value")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
