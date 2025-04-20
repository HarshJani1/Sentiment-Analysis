# Sentiment Analysis Pipeline


## Overview
This project implements an end-to-end **Sentiment Analysis** pipeline on text data, comparing multiple machine learning and deep learning models for binary sentiment classification (positive vs. negative). It covers data ingestion, advanced preprocessing (including negation handling and lemmatization), feature extraction with TF‑IDF, data balancing with SMOTE, model training (traditional, ensemble, and neural networks), evaluation, and visualization of performance metrics.

---

## Key Features

- **Data Preprocessing**
  - Cleans text by removing non-alphabetic characters and lowercasing.
  - **Negation handling**: Marks words following negation terms (`not`, `no`, `never`, `nor`) with a `_NOT` suffix.
  - **Lemmatization** using NLTK's `WordNetLemmatizer`.
  - Stop‑word removal (preserving negations).

- **Feature Extraction**
  - TF‑IDF vectorization with enhanced settings:
    - `ngram_range=(1,3)` to include unigrams, bigrams, and trigrams.
    - `sublinear_tf=True`, `min_df=3`, `max_df=0.9`, and built‑in English stop‑words.
    - Increased `max_features=5000` for richer vocabulary.

- **Data Balancing**
  - Addresses class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).

- **Modeling**
  - **Support Vector Machine** (SVM) with extensive grid search on `C`, `kernel`, `gamma`, and `class_weight`.
  - **Random Forest** with grid search on `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight`.
  - **Multinomial Naive Bayes** with grid search on `alpha` and `fit_prior`.
  - **Voting Classifier** (soft voting ensemble of SVM, RF, and NB).
  - **Stacking Classifier** (meta‑learning ensemble of SVM and RF with a logistic regression final estimator).
  - **Neural Network**: Multi‑layer perceptron built with TensorFlow/Keras (512→256→128 units with Dropout).

- **Evaluation & Visualization**
  - Metrics: **Accuracy**, **Precision**, **Recall**, **F1 Score**, **Confusion Matrix** for all models.
  - **Training history** plots for the neural network (accuracy & loss over epochs).
  - Bar charts comparing model accuracy and a combined chart of all performance metrics.

- **Model Persistence**
  - Saves the best stacking classifier (`best_model.pkl`) and the TF‑IDF vectorizer (`tfidf_vectorizer.pkl`).

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/Sentiment-Analysis-Advanced.git
    cd Sentiment-Analysis-Advanced
    ```
2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Prepare your data**: Place your CSV file(s) in the `data/` directory. Ensure columns `text` and `sentiment` (0 = negative, 1 = positive).
2. **Run the pipeline**:
    ```bash
    python main.py
    ```
3. **Inspect outputs**:
   - Console logs show detailed confusion matrices and metrics.
   - PNG charts are saved to `output/`.
   - Serialized models and vectorizer are saved at project root.

---

## Comparison with Previous Version

| Aspect                        | Previous Pipeline                                          | Enhanced Pipeline (This Version)                                                |
|-------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------|
| Preprocessing                 | PorterStemmer + basic stop‑word removal                    | Negation tagging + WordNet lemmatization + refined stop‑words                   |
| TF‑IDF                        | `max_features=1500`, unigrams only                        | `max_features=5000`, n‑grams up to trigrams, sublinear TF, `min_df`/`max_df`     |
| Data Split                    | Random split                                              | Stratified split to preserve class distribution                                   |
| Class Imbalance               | Not addressed                                             | **SMOTE** oversampling                                                           |
| Traditional Models            | SVM, Random Forest, Naive Bayes                            | Same models with richer hyperparameter grids                                     |
| Ensembling                    | Voting Classifier                                          | Voting + **Stacking Classifier**                                                  |
| Deep Learning                 | —                                                          | **Neural Network** (multi‑layer perceptron with Dropout, trained for 30 epochs)  |
| Evaluation Metrics            | Accuracy, Precision, Recall, MAE, MSE, RMSE                | Accuracy, Precision, Recall, **F1 Score**, Confusion Matrix                      |
| Visualizations                | Two bar charts (accuracy vs. error metrics)                | Bar charts with annotations + combined metrics plot + NN training curves        |
| Model Persistence             | X (no serialization)                                       | Saves best stacking model and TF‑IDF vectorizer as `.pkl` files                  |

---

## Detailed Performance Metrics

### Previous Pipeline

| Model               | Accuracy | Precision | Recall | MAE   | MSE   | RMSE  |
|---------------------|----------|-----------|--------|-------|-------|-------|
| SVM                 | 0.856    | 0.856     | 0.856  | 0.144 | 0.144 | 0.379 |
| Random Forest       | 0.832    | 0.832     | 0.832  | 0.168 | 0.168 | 0.410 |
| Naive Bayes         | 0.850    | 0.850     | 0.850  | 0.150 | 0.150 | 0.387 |
| Voting Classifier   | 0.851    | 0.851     | 0.851  | 0.149 | 0.149 | 0.386 |
| K-Means (baseline)  | 0.474    | 0.473     | 0.474  | 0.526 | 0.526 | 0.726 |

### Enhanced Pipeline

| Model                 | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| SVM                   | 0.8640   | 0.8648    | 0.8640 | 0.8644   |
| Random Forest         | 0.8330   | 0.8352    | 0.8330 | 0.8341   |
| Naive Bayes           | 0.8570   | 0.8571    | 0.8570 | 0.8571   |
| Voting Classifier     | 0.8700   | 0.8705    | 0.8700 | 0.8702   |
| Stacking Classifier   | 0.8690   | 0.8694    | 0.8690 | 0.8692   |
| Neural Network        | 0.8350   | 0.8351    | 0.8350 | 0.8350   |

### Old :
---
![Model Accuracy Comparison](https://github.com/user-attachments/assets/5d3efd24-8bfa-46f1-a176-e7a4329ba845)
![Model Error Comparison](https://github.com/user-attachments/assets/b94e6287-c9f3-4ca6-a552-9b1dd5e63b27)
### New :
---
<img width="100%" alt="Screenshot 2025-04-20 at 3 09 29 PM" src="https://github.com/user-attachments/assets/10f077d3-805e-47d5-9e94-1e13c58366e0" />
<img width="100%" alt="Screenshot 2025-04-20 at 3 09 15 PM" src="https://github.com/user-attachments/assets/7e82a1f6-f08b-4a71-8ad8-1da41bc8a88b" />



---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy Coding!*

