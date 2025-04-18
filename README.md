# Sentiment Analysis Project
![Model Accuracy Comparison](https://github.com/user-attachments/assets/5d3efd24-8bfa-46f1-a176-e7a4329ba845)
![Model Error Comparison](https://github.com/user-attachments/assets/b94e6287-c9f3-4ca6-a552-9b1dd5e63b27)




## Overview
This project implements a Sentiment Analysis pipeline on text data, comparing multiple machine learning models for binary sentiment classification (positive vs. negative). The pipeline covers data ingestion, preprocessing, feature extraction using TF-IDF, model training, evaluation, and visualization of performance metrics.

## Features
- **Data Preprocessing**: Cleaning, tokenization, stop-word removal, and lemmatization.
- **Feature Extraction**: TF-IDF vectorization to convert text into numerical representations.
- **Modeling**: Comparison of five models:
  1. Support Vector Machine (SVM)
  2. Random Forest Classifier
  3. Multinomial Naive Bayes
  4. Voting Classifier (ensemble of SVM, Random Forest, and Naive Bayes)
  5. K-Means Clustering (baseline unsupervised approach)
- **Evaluation Metrics**: Accuracy, Precision, Recall, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Confusion Matrices.
- **Visualizations**: Bar charts comparing error metrics and accuracy across models.

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/HarshJani1/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```
2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare your data**: Place your CSV files in the `data/` directory. Ensure they follow the format: `text`, `sentiment`.
2. **Run the main script**:
   ```bash
   python main.py
   ```
3. **View results**:
   - Evaluation metrics and confusion matrices will be printed to the console.
   - Charts are saved to `output/` as PNG files.

## Dataset
- The sample dataset consists of labeled text reviews with binary sentiment labels (`0` = Negative, `1` = Positive).
- Feel free to replace with your own dataset; update the path in `config.py` or modify `main.py` accordingly.

## Preprocessing
- **Text Cleaning**: Removal of URLs, punctuation, and non-alphanumeric characters.
- **Tokenization**: Splitting text into tokens.
- **Stop-Word Removal**: Filtering out common stop words (e.g., "the", "and").
- **Lemmatization**: Converting tokens to their lemma form.

## Models & Performance
| Model               | Accuracy | Precision | Recall | MAE   | MSE   | RMSE  |
|---------------------|----------|-----------|--------|-------|-------|-------|
| SVM                 | 0.856    | 0.856     | 0.856  | 0.144 | 0.144 | 0.379 |
| Random Forest       | 0.832    | 0.832     | 0.832  | 0.168 | 0.168 | 0.410 |
| Naive Bayes         | 0.850    | 0.850     | 0.850  | 0.150 | 0.150 | 0.387 |
| Voting Classifier   | 0.851    | 0.851     | 0.851  | 0.149 | 0.149 | 0.386 |
| K-Means (baseline)  | 0.474    | 0.473     | 0.474  | 0.526 | 0.526 | 0.726 |

Refer to the embedded charts above for a visual comparison of these metrics.

## Project Structure
```
├── assets/
│   ├── error_metrics_comparison.jpg
│   └── accuracy_comparison.jpg
├── data/
│   └── ... (CSV datasets)
├── output/
│   └── ... (generated charts and logs)
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
├── requirements.txt
├── README.md
└── config.py
```

## Contributing
Contributions are welcome! To propose changes:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Commit your changes and push: `git push origin feature/YourFeature`.
4. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy Coding!*

