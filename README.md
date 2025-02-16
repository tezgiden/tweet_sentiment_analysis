# Tweet Sentiment Analysis

This project implements a **Tweet Sentiment Analysis** tool using **Support Vector Machines (SVM)**. It was developed as part of the **Columbia University AI Certificate Program** under the **Natural Language Processing (NLP)** course. The tool analyzes the sentiment of tweets, classifying them into three categories: **negative**, **neutral**, and **positive**.

The project incorporates advanced NLP techniques, including **n-gram analysis**, **lexicon-based features**, **linguistic features**, and a **custom negation feature**. The implementation is based on research from the paper: [Sentiment Analysis of Short Informal Texts](https://aclanthology.org/S13-2053.pdf).

---

## Features and Models

The project implements several models and features to analyze tweet sentiment:

### 1. **N-Gram Model**
- Utilizes `TfidfVectorizer` to extract **unigrams** and **bigrams** from the tweet dataset.
- Transforms text data into numerical features for training the SVM model.

### 2. **Lexicon-Based Features**
- Incorporates sentiment scores from **unigrams** and **bigrams** using predefined lexicons (`Hashtag` and `Sentiment140`).
- Extracts the following features:
  1. Total count of tokens with a positive score > 0.
  2. Total count of tokens with a negative score > 0.
  3. Summed positive score of all tokens.
  4. Summed negative score of all tokens.
  5. Maximum positive score of all tokens.
  6. Maximum negative score of all tokens.
  7. Maximum positive score of the last token/bigram.
  8. Maximum negative score of the last token/bigram.

### 3. **Linguistic Features**
- Extracts **Part-of-Speech (POS)** tags and other linguistic patterns:
  1. Number of tokens with all characters capitalized.
  2. Counts of specific POS tags (e.g., `!`, `#`, `$`, `@`, `A`, `D`, etc.).
  3. Number of hashtags.
  4. Number of words with a character repeated more than twice.

### 4. **Custom Feature (Negation)**
- Analyzes the presence of **negation sentences** in tweets.
- Identifies sentences starting with negative words (e.g., "never", "no", "not") and ending with punctuation marks.

---

## Implementation Details

### Code Overview
The project is implemented in Python and leverages the following libraries:
- **Scikit-learn**: For SVM, feature extraction (`TfidfVectorizer`), and evaluation metrics.
- **NLTK**: For tokenization, stopword removal, and lemmatization.
- **Gensim**: For Word2Vec (experimental, not used in the final model).
- **Emoji**: For handling emojis in tweets.

### Key Functions
1. **Data Loading and Preprocessing**:
   - `load_datafile()`: Loads and preprocesses tweet data from CSV files.
   - `clean_tweet()`: Cleans tweets by removing URLs, mentions, stopwords, and performing lemmatization.
   - `light_clean_tweet()`: Light preprocessing for retaining certain tweet characteristics.

2. **Feature Extraction**:
   - `extract_ngram_features()`: Extracts n-gram features using `TfidfVectorizer`.
   - `extract_lexicon_based_features()`: Computes sentiment-based features using lexicons.
   - `extract_linguistic_features()`: Extracts linguistic features (POS tags, hashtags, etc.).
   - `extract_negation_features()`: Identifies negation sentences in tweets.

3. **Model Training and Evaluation**:
   - `train_and_evaluate()`: Trains the SVM model using `RandomizedSearchCV` for hyperparameter tuning and evaluates its performance.
   - `analyze_predictions()`: Analyzes misclassified and correctly classified examples for model debugging.

4. **Utility Functions**:
   - `load_sentiment_lexicon()`: Loads sentiment lexicons for unigrams and bigrams.
   - `extract_features()`: Combines all features (n-gram, lexicon, linguistic, and custom) for training and evaluation.

---

## Results

The performance of the models was evaluated using **f1-score accuracy**. The results are as follows:

| Model                     | f1-score Accuracy |
|---------------------------|-------------------|
| N-Gram                    | 0.5238            |
| N-Gram + Lexicon          | 0.5488            |
| N-Gram + Lexicon + Linguistic | 0.5417        |
| N-Gram + Lexicon + Linguistic + Custom | 0.5476 |

The **best-performing model** was **N-Gram + Lexicon + Linguistic + Custom**, achieving an **f1-score accuracy of 0.5476**.

### Classification Report
|          |precision|   recall  |f1-score   |support|

|negative     |0.3503    |0.4276    |0.3851       145|
 neutral     0.5806    0.6136    0.5967       352
positive     0.6254    0.5306    0.5741       343
accuracy                         0.5476       840
macro avg    0.5188    0.5239    0.5186       840
weighted avg 0.5592    0.5476    0.5510       840

---

## Usage

To run the project, use the following command:

```bash
python Solution-2.py --model <model_name> --lexicon <lexicon_name> --train <train_filepath> --evaluation <evaluation_filepath>
```

## Arguments
--model: The model to train and evaluate. Options: Ngram, Lex, Ling, Ngram+Lex, Ngram+Ling, Ngram+Lex+Ling, Ngram+custom, custom, Ngram+Lex+Ling+custom.

--lexicon: The sentiment lexicon to use. Options: Hashtag, Sentiment140.

--train: Path to the training dataset (CSV file).

--evaluation: Path to the evaluation dataset (CSV file).

## Example
```bash
python Solution-2.py --model Ngram+Lex+Ling+custom --lexicon Hashtag --train ./data/train.csv --evaluation ./data/dev.csv
```

## Dependencies
To install the required dependencies, run:
pip install -r requirements.txt

## Future Work
Experiment with deep learning models (e.g., LSTM, BERT) for improved accuracy.

Incorporate additional lexicons and domain-specific sentiment analysis.

Optimize feature extraction and model training for real-time sentiment analysis.

Acknowledgments
Columbia University AI Certificate Program for providing the foundation for this project.

ACL Anthology for the research paper that inspired the feature implementation.
