import numpy as np
import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_absolute_error, \
    mean_squared_error
# from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
dataset = pd.read_csv('./Test.csv')
print(f"Dataset size: {len(dataset)}")

# Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
corpus = []
all_stopwords = stopwords.words('english')
# Keep important negation words
negation_words = {'not', 'no', 'nor', 'never'}
all_stopwords = [word for word in all_stopwords if word not in negation_words]

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    # Handle negation by appending _NOT
    words = review.split()
    negate = False
    processed = []
    for word in words:
        if word in negation_words:
            negate = True
            processed.append(word)
            continue
        if negate:
            word += "_NOT"
            negate = False
        processed.append(word)
    review = ' '.join(processed)
    # Lemmatization and stopword removal
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

print('corpus done')
# TF-IDF with enhanced parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Include trigrams
    sublinear_tf=True,
    min_df=3,
    max_df=0.9,
    stop_words='english'
)
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21, stratify=y)
print('Split done')
# Handle class imbalance with SMOTE
smote = SMOTE(random_state=21)
X_train, y_train = smote.fit_resample(X_train, y_train)
print('SMOTE done')
# SVM with enhanced GridSearch
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'class_weight': ['balanced', None]
}
svm = GridSearchCV(SVC(probability=True, random_state=21), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm.fit(X_train, y_train)
svm_best = svm.best_estimator_

print("svm done")

# Random Forest with enhanced GridSearch
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}
rf = GridSearchCV(RandomForestClassifier(random_state=21), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf.fit(X_train, y_train)
rf_best = rf.best_estimator_
print("rf done")
# Naive Bayes with enhanced GridSearch
nb_param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}
nb = GridSearchCV(MultinomialNB(), nb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
nb.fit(X_train, y_train)
nb_best = nb.best_estimator_
print("nb done")
# XGBoost Classifier
# xgb_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'scale_pos_weight': [1, 5, 10]
# }
# xgb = GridSearchCV(XGBClassifier(random_state=21, eval_metric='mlogloss'), xgb_param_grid, cv=5, n_jobs=-1)
# xgb.fit(X_train, y_train)
# xgb_best = xgb.best_estimator_

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
print("voting done")
# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf', rf_best),
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
print("stacking done")
# Neural Network implementation
le = LabelEncoder()
y_train_nn = le.fit_transform(y_train)
y_test_nn = le.transform(y_test)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train_nn,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
print("CNN done")
# K-Means clustering (for comparison)
# kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=21, n_init=10)
# kmeans.fit(X_train)
# kmeans_pred = kmeans.predict(X_test)

# Store evaluation results
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}


# Enhanced evaluation function
def evaluate_model(name, model, y_true, y_pred=None, is_nn=False):
    if is_nn:
        y_pred = np.argmax(model.predict(X_test), axis=1)
    elif y_pred is None:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)

    print(f"\n--- {name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)
    results["F1"].append(f1)


# Evaluate all models
evaluate_model("SVM", svm_best, y_test)
evaluate_model("Random Forest", rf_best, y_test)
evaluate_model("Naive Bayes", nb_best, y_test)
# evaluate_model("XGBoost", xgb_best, y_test)
evaluate_model("Voting Classifier", voting_clf, y_test)
evaluate_model("Stacking Classifier", stacking_clf, y_test)
evaluate_model("Neural Network", model, y_test_nn, is_nn=True)
# evaluate_model("K-Means", kmeans, y_test, kmeans_pred)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plotting
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")
ax = sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title("Model Accuracy Comparison", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

# Plot comprehensive metrics
metrics_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 8))
sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_df, palette='muted')
plt.title("Model Performance Metrics", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot neural network training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Save the best model (Stacking Classifier)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)

# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)