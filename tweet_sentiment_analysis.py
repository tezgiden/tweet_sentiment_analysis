import os
import csv
import argparse
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add this import at the top of the file with other imports
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # Import tqdm for progress tracking
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np
from gensim.models import Word2Vec

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

LEXICON_DIRS = {
    'Hashtag': 'lexica/Hashtag-Sentiment-Lexicon',
    'Sentiment140': 'lexica/Sentiment140-Lexicon'
}
####################################################################################
######################====> R E A D   M E!!!!! <====################################
####################################################################################

#----------------IMPORTANT!!!!---------------------

# 1-) I developed this code using the Cursor which utilizes code-complete as well as the 
# AI to suggest, improve and fix the code.
# 2-) I used DeepSeek to lookup RandomizedSearchCV to train and test various hyperparameters
# 3-) I left some printouts active to see the test resutls
# 4-) I comented out some of the debugging print statements as is. 
# 5-) I changed few method signatures other than the main method
# 6-) I tried few variation to cleanup the tweets in clean_tweet(). So far what is left in this method performed best. 
# 7-) I tried to use Word2Vec for the Ngrams but TfidfVectorizer performed better. I still left Word2Vec code in here as commented out.
####################################################################################
####################################################################################


def load_datafile(filepath):
    data = []
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):    
            label = row['label'];  
            if 'objective' != label.lower():
                tweet = light_clean_tweet(row['tweet_tokens'])      
                cleaned_tweet = clean_tweet(row['tweet_tokens'])
                lowercase_tweet = tweet.lower()
                data.append({
                    'tokens': tweet.split(),
                    'pos_tags': row['pos_tags'].split(),
                    'label': row['label'],
                    'cleaned_tokens': cleaned_tweet.split(),
                    'lowercase_tokens': lowercase_tweet.split(),
                    'tweet': lowercase_tweet
                })
    return data
    

def load_sentiment_lexicon(lexicon):
    unigram_scores = {}
    with open(os.path.join(LEXICON_DIRS[lexicon], 'unigrams.txt'), 'r') as f:
        for line in f.readlines():
            unigram, score, _, __ = line.strip().split('\t')
            unigram_scores[unigram] = {
                'pos_score': float(score),
                'neg_score': -1 * float(score)
            }
    
    bigram_scores = {}
    with open(os.path.join(LEXICON_DIRS[lexicon], 'bigrams.txt'), 'r') as f:
        for line in f.readlines():
            bigram, score, _, __ = line.strip().split('\t')
            bigram_scores[tuple(bigram.split(' '))] = {
                'pos_score': float(score),
                'neg_score': -1 * float(score)
            }
    
    return unigram_scores, bigram_scores


def clean_tweet(tweet):
    # I tested with few cleanup methods but it looks like keeping some of those performed better
    # Remove URLs
    #tweet = re.sub(r'http\S+|www\S+|https\S+', 'http\:\/\/someurl', tweet, flags=re.MULTILINE)
    # Remove mentions 
    #tweet = re.sub(r'@\w+', '@someone', tweet)
    # Remove special characters and punctuation
    #tweet = re.sub(r'[^\w\s]', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    # Tokenization
    tokens = word_tokenize(tweet)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove numbers
    # tweet = re.sub(r'\d+', '', tweet)
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    # Handle emojis
    tweet = emoji.demojize(tweet)
    return tweet


def light_clean_tweet(tweet):
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    # Handle emojis
    tweet = emoji.demojize(tweet)
    return tweet

# Using the specified sklearn vectorizer, trains the vectorizer on the train_set 
# and returns vectorized representations of train_set, evaluation_set
#
# Input: 
#   vectorizer - an sklearn vectorizer (Count, Tfidf) instance
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#
# hint: https://stackoverflow.com/questions/48671270/use-sklearn-tfidfvectorizer-with-already-tokenized-inputs
def extract_ngram_features(vectorizer, train_set, evaluation_set):
    # for ngrams I used cleaned_tokens to match across tweets where it makes sense
    train_texts = [' '.join(item['cleaned_tokens']) for item in train_set]
    evaluation_texts = [' '.join(item['cleaned_tokens']) for item in evaluation_set]
    # print("train_texts", train_texts[0]) #=> train_texts Plant City is babysitting NYC palm trees for the winter . Crap like this cracks me up ! http://t.co/BmJAFvCA
    vectorizer.fit(train_texts)
    train_features = vectorizer.transform(train_texts)
    evaluation_features = vectorizer.transform(evaluation_texts)
    # print("train_features", train_features[0]) #==>  (0, 77538)     0.20481392266832113 ==>   (0, 77535)    0.16364306721634944  ==>   (0, 73550)    0.19549718923203502 total of 35
    
    return train_features, evaluation_features

# Using the specified unigram_scores and bigram_scores, 
# extracts 4 lexicon based features for specified train_set and evaluation_set
# 
# Input: 
#   unigram_scores - a dictionary as returned by load_sentiment_lexicon 
#   bigram_scores - a dictionary as returned by load_sentiment_lexicon
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#       where each representation is of dimension (# documents x 8)
#   
#       Please encode the lexicon based features in the following order:
#           0 - total count of tokens (unigrams + bigrams) in the tweet with positive score > 0
#           1 - total count of tokens (unigrams + bigrams) in the tweet with negative score > 0
#           2 - summed positive score of all the unigrams and bigrams in the tweet 
#           3 - summed negative score of all the unigrams and bigrams in the tweet
#           4 - the max positive score of all the unigrams and bigrams in the tweet
#           5 - the max negative score of all the unigrams and bigrams in the tweet
#           6 - the max of the positive scores of the last unigram / bigram in the tweet (with score > 0)
#           7 - the max of the negative scores of the last unigram / bigram in the tweet (with score > 0)    
def extract_lexicon_based_features(unigram_scores, bigram_scores, train_set, evaluation_set):
    # Helper function to compute lexicon-based features for a single document (tweet).
    def get_features(tokens):
        # Initialize variables to store the computed features:
        # - pos_count: Number of tokens with a positive sentiment score > 0.
        # - neg_count: Number of tokens with a negative sentiment score > 0.
        # - pos_sum: Sum of all positive sentiment scores in the document.
        # - neg_sum: Sum of all negative sentiment scores in the document.
        # - max_pos: Maximum positive sentiment score in the document.
        # - max_neg: Maximum negative sentiment score in the document.
        # - last_pos: Maximum positive sentiment score of the last token/bigram in the document.
        # - last_neg: Maximum negative sentiment score of the last token/bigram in the document.
        pos_count = 0
        neg_count = 0
        pos_sum = 0.0
        neg_sum = 0.0
        max_pos = 0.0
        max_neg = 0.0
        last_pos = 0.0
        last_neg = 0.0
        
        # Iterate through each token in the document.
        for i in range(len(tokens)):
            token = tokens[i]
            
            # Check if the token exists in the unigram sentiment lexicon.
            if token in unigram_scores:
                # Update the sum of positive and negative sentiment scores.
                pos_score = unigram_scores[token]['pos_score']
                pos_sum += pos_score
                neg_score = unigram_scores[token]['neg_score']
                neg_sum += neg_score
                #if pos
                # Update the count of positive and negative tokens.
                if pos_score > 0:
                    pos_count += 1
                    last_pos = max(last_pos, pos_score)
                if neg_score > 0:
                    neg_count += 1
                    last_neg = neg_score
                
                # Update the maximum positive and negative sentiment scores.
                max_pos = max(max_pos, pos_score)
                max_neg = max(max_neg, neg_score)
                            
            # Check if the current token and the next token form a bigram in the bigram sentiment lexicon.
            if i < len(tokens) - 1:
                bigram = tuple(tokens[i:i+2])  # Create a bigram tuple.
                if bigram in bigram_scores:
                    # print("get_features: bigram match:", bigram)
                    # Update the sum of positive and negative sentiment scores for the bigram.
                    pos_score = bigram_scores[bigram]['pos_score']
                    pos_sum += pos_score
                    neg_score = bigram_scores[bigram]['neg_score']
                    neg_sum += neg_score
                    #if pos
                    # Update the count of positive and negative tokens.
                    if pos_score > 0:
                        pos_count += 1
                        last_pos = max(last_pos, pos_score)
                    if neg_score > 0:
                        neg_count += 1
                        last_neg = neg_score
                    
                    # Update the maximum positive and negative sentiment scores.
                    max_pos = max(max_pos, pos_score)
                    max_neg = max(max_neg, neg_score)
                                        
        # print("get_features: Tokens=", tokens)
        # print("get_features: pos_count=", pos_count, ", neg_count:",neg_count, ", pos_sum:", pos_sum, ", neg_sum:", neg_sum, ", max_pos:", max_pos, ", max_neg:", max_neg, ", last_pos:", last_pos, ", last_neg:", last_neg)
        # Return the computed features as a list.
        return [pos_count, neg_count, pos_sum, neg_sum, max_pos, max_neg, last_pos, last_neg]
    
    # Compute lexicon-based features for all documents in the training set.
    train_features = [get_features(item['lowercase_tokens']) for item in train_set]
    
    # print("extract_lexicon_based_features: train_features=", train_features[0])
    # print("extract_lexicon_based_features: train_set=", train_set[0])
    # print("extract_lexicon_based_features: train_features-1=", train_features[1])
    # print("extract_lexicon_based_features: train_set-1=", train_set[1])
    
    # Compute lexicon-based features for all documents in the evaluation set.
    evaluation_features = [get_features(item['lowercase_tokens']) for item in evaluation_set]
    
    # Convert the lists of features into sparse matrices for efficient storage and computation.
    # `csr_matrix` is used to represent the feature matrices in Compressed Sparse Row format.
    return csr_matrix(train_features), csr_matrix(evaluation_features)

# Extract the 4 linguistic features for specified train_set and evaluation_set. 
#
# Input: 
#   train_set - an array of dictionaries as returned by load_datafile
#   evaluation_set - an array of dictionaries as returned by load_datafile
#
# Output:
#   A 2-tuple: (vectorized train_set representations, vectorized evaluation_set representations)
#       where each representation is of dimension (# documents x 26)
#
#       Please encode the linguistic features in the following order:
#           0 - number of tokens with all their characters capitalized
#           1-23 - separate counts of each POS tag in the following sorted order:
#               [
#                   '!', '#', '$', '&', ',', '@', 'A', 'D', 'E', 'G', 'L', 'N', 
#                   'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', '^', '~'
#               ]
#           24  - number of hashtags
#           25  - number of words with one character repeated more than two times
def extract_linguistic_features(train_set, evaluation_set):
    # Define the order of POS tags to ensure consistent feature indexing.
    # These tags represent specific linguistic categories (e.g., punctuation, nouns, verbs, etc.).
    pos_tags_order = ['!', '#', '$', '&', ',', '@', 'A', 'D', 'E', 'G', 'L', 'N', 
                      'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', '^', '~']
    
    # Helper function to compute linguistic features for a single document (tweet).
    def get_features(tokens, pos_tags):
        # Initialize a list of 26 features with default value 0.
        # Each index in the list corresponds to a specific linguistic feature.
        features = [0] * 26

        # Feature 0: Count of tokens where all characters are uppercase.
        features[0] = sum(1 for token in tokens if token.isupper())
        
        # Initialize a dictionary to count occurrences of each POS tag in the document.
        pos_counts = {tag: 0 for tag in pos_tags_order}
        
        # Iterate through the POS tags in the document and update their counts.
        for tag in pos_tags:
            if tag in pos_counts:
                pos_counts[tag] += 1
        
        # Features 1-23: Populate the feature list with counts of each POS tag.
        # The order of POS tags is determined by `pos_tags_order`.
        for i, tag in enumerate(pos_tags_order):
            features[i+1] = pos_counts[tag]
        
        # Feature 24: Count of hashtags in the document (tokens starting with '#').
        features[24] = sum(1 for token in tokens if token.startswith('#'))
        
        # Feature 25: Count of tokens with a character repeated more than two times.
        # For example, "loooove" has 'o' repeated more than twice.
        features[25] = sum(1 for token in tokens if any(token.count(c) > 2 for c in set(token)))
        
        # print("get_features: tokens:", tokens)
        # for j, feature in enumerate(features):
        #     print(f"{feature}: {j}")
        # Return the computed features for the document.
        return features
    
    # Compute linguistic features for all documents in the training set.
    # `get_features` is applied to each document in `train_set`.
    train_features = [get_features(item['tokens'], item['pos_tags']) for item in train_set]
    
    # Compute linguistic features for all documents in the evaluation set.
    # `get_features` is applied to each document in `evaluation_set`.
    evaluation_features = [get_features(item['tokens'], item['pos_tags']) for item in evaluation_set]
    
    # Convert the lists of features into sparse matrices for efficient storage and computation.
    # `csr_matrix` is used to represent the feature matrices in Compressed Sparse Row format.
    return csr_matrix(train_features), csr_matrix(evaluation_features)
    
def extract_negation_features(train_set, evaluation_set):    
    # List of negative words to check at the beginning of a sentence
    negative_words = [
        "never", "no", "nothing", "nowhere", "noone", "none", "not",
        "havent", "hasnt", "hadnt", "cant", "couldnt", "shouldnt",
        "wont", "wouldnt", "dont", "doesnt", "didnt", "isnt", "arent", "aint",
        "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldnt'",
        "won't", "wouldn't", "don't", "doesn't", "didn't", "isn't", "aren't", "ain't"
    ]

    # List of punctuation marks to check at the end of a sentence
    punctuation_marks = [',', '.', ':', ';', '!', '?']
    
    # Regular expression pattern to match sentences starting with negative words and ending with punctuation
    pattern = re.compile(rf'^\b({"|".join(negative_words)})\b.*[{re.escape("".join(punctuation_marks))}]$', re.IGNORECASE)

    def check_sentence(sentence):
        """Check if a sentence starts with a negative word and ends with a punctuation mark."""
        return bool(pattern.match(sentence.strip()))

    def analyze_tweet(tweet):
        
        features = [0]
        """Analyze a tweet to find sentences that match the criteria."""
        sentences = re.split(r'(?<=[.!?;:])\s+', tweet)  # Split tweet into sentences
        matching_sentences = []
        #print("sentences=", sentences)
        for sentence in sentences:
            if check_sentence(sentence):
                matching_sentences.append(sentence)
        features[0] = len(matching_sentences)
        if len(matching_sentences)>0:
            print("analyze_tweet=", tweet, ", ===> matching_sentences=", matching_sentences)
        return features
    
    train_features = [analyze_tweet(item['tweet']) for item in train_set]
    
    # print("extract_negation_features: train_features=", train_features[0])
    # print("extract_negation_features: train_set=", train_set[0])
    # print("extract_negation_features: train_features-1=", train_features[1])
    # print("extract_negation_features: train_set-1=", train_set[1])
    
    # Compute lexicon-based features for all documents in the evaluation set.
    evaluation_features = [analyze_tweet(item['tweet']) for item in evaluation_set]
    
    # Convert the lists of features into sparse matrices for efficient storage and computation.
    # `csr_matrix` is used to represent the feature matrices in Compressed Sparse Row format.
    return csr_matrix(train_features), csr_matrix(evaluation_features)

# Train Word2Vec model <========= THis didn't work as well as TfidfVectorizer
# def train_word2vec(train_set, vector_size=100, window=5, min_count=1, workers=4):
#     # Extract tokens from the training set
#     sentences = [item['cleaned_tokens'] for item in train_set]
    
#     # Train Word2Vec model
#     print("Training Word2Vec model...")
#     word2vec_model = Word2Vec(
#         sentences=sentences,
#         vector_size=vector_size,  # Dimensionality of the word vectors
#         window=window,            # Maximum distance between the current and predicted word
#         min_count=min_count,      # Ignores words with frequency lower than this
#         workers=workers           # Number of CPU cores
#     )
#     return word2vec_model

# # Vectorize tweet tokens using Word2Vec
# def vectorize_tokens(word2vec_model, tokens):
#     vectors = []
#     for token in tokens:
#         if token in word2vec_model.wv:  # Check if the token is in the Word2Vec vocabulary
#             vectors.append(word2vec_model.wv[token])  # Get the word vector
#     if len(vectors) > 0:
#         return np.mean(vectors, axis=0)  # Average the word vectors
#     else:
#         return np.zeros(word2vec_model.vector_size)  # Return a zero vector if no tokens are found

# Extract Word2Vec features
# def extract_word2vec_features(word2vec_model, dataset):
#     features = []
#     for item in tqdm(dataset, desc="Extracting Word2Vec features"):
#         tokens = item['cleaned_tokens']
#         feature_vector = vectorize_tokens(word2vec_model, tokens)
#         features.append(feature_vector)
#     return np.array(features)

#   Extracts training and validation features as specified by the model.
#
#   Returns a 4-tuple of:
#       0 - training features (# of train documents x # of features)
#       1 - training labels (# of train documents)
#       2 - evaluation features (# of evaluation documents x # of features)
#       3 - evaluation labels (# of evaluation documents)
#
#       When encoding labels, please use the following mapping:
#           'negative' => 0
#           'neutral' => 1
#           'objective' => 2
#           'positive' => 3
def extract_features(model, lexicon, train_set, evaluation_set):
    # Load sentiment lexicon (unigram and bigram scores) for the specified lexicon type.
    # The lexicon is used to compute sentiment-based features.
    unigram_scores, bigram_scores = load_sentiment_lexicon(lexicon)

    # Initialize empty sparse matrices for training and evaluation features.
    # These matrices will store the combined features for all documents.
    # The number of rows is equal to the number of documents, and the number of columns starts at 0.
    train_features = csr_matrix((len(train_set), 0))
    evaluation_features = csr_matrix((len(evaluation_set), 0))
    print("train_features", train_features.shape)  

    # Extract n-gram features if the model includes 'ngram'.
    if 'ngram' in model.lower():
        # Initialize a TfidfVectorizer to compute n-gram features (unigrams and bigrams).
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        # word2vec didin't perform well. So I stick with  TfidfVectorizer instead
        # Train Word2Vec model on the training set
        # word2vec_model = train_word2vec(train_set)
        # # Extract Word2Vec features for training and evaluation sets
        # train_ngram_feats = extract_word2vec_features(word2vec_model, train_set)
        # evaluation_ngram_feats = extract_word2vec_features(word2vec_model, evaluation_set)
        
        # train_features = hstack([train_features, train_ngram_feats])
        # evaluation_features = hstack([evaluation_features, evaluation_ngram_feats])
        
        # # Extract n-gram features for the training and evaluation sets.
        train_ngram_feats, evaluation_ngram_feats = extract_ngram_features(vectorizer, train_set, evaluation_set)
        
        # # Append the n-gram features to the existing feature matrices using horizontal stacking.
        train_features = hstack([train_features, train_ngram_feats])
        evaluation_features = hstack([evaluation_features, evaluation_ngram_feats])
        print("extract_features DONE ngram ")  
    # Extract lexicon-based features if the model includes 'lex'.
    if 'lex' in model.lower():
        # Compute lexicon-based features using the loaded unigram and bigram scores.
        train_lexicon_feats, evaluation_lexicon_feats = extract_lexicon_based_features(unigram_scores, bigram_scores, train_set, evaluation_set)
        
        # Append the lexicon-based features to the existing feature matrices.
        train_features = hstack([train_features, train_lexicon_feats])
        evaluation_features = hstack([evaluation_features, evaluation_lexicon_feats])
        print("extract_features DONE lex ")  
    
    # Extract linguistic features if the model includes 'ling'.
    if 'ling' in model.lower():
        # Compute linguistic features (e.g., POS tag counts, hashtags, etc.).
        train_linguistic_feats, evaluation_linguistic_feats = extract_linguistic_features(train_set, evaluation_set)
        
        # Append the linguistic features to the existing feature matrices.
        train_features = hstack([train_features, train_linguistic_feats])
        evaluation_features = hstack([evaluation_features, evaluation_linguistic_feats])
        print("extract_features DONE lex ")  
    
    # If the model is 'Custom', implement additional custom features here.
    if 'custom' in model.lower():
        train_negation_feats, evaluation_negation_feats = extract_negation_features(train_set, evaluation_set)
        # Append the linguistic features to the existing feature matrices.
        train_features = hstack([train_features, train_negation_feats])
        evaluation_features = hstack([evaluation_features, evaluation_negation_feats])
        print("extract_features DONE custom ")  

    # Map the string labels (e.g., 'negative', 'positive') to numerical values for training and evaluation.
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    Y_train = [label_mapping[item['label']] for item in train_set]  # Training labels
    Y_test = [label_mapping[item['label']] for item in evaluation_set]  # Evaluation labels

    # Return the feature matrices and corresponding labels for training and evaluation.
    return train_features, Y_train, evaluation_features, Y_test
    

def analyze_predictions(evaluation_set, Y_test, Y_pred):
    """
    Analyze successful and failed predictions.
    """
    # Map numerical labels back to their string names
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Store misclassified and correctly classified examples
    misclassified = []
    correct = []

    for i, (true_label, pred_label) in enumerate(zip(Y_test, Y_pred)):
        tweet = evaluation_set[i]['tokens']  # Get the tweet text
        true_label_str = label_mapping[true_label]
        pred_label_str = label_mapping[pred_label]

        if true_label != pred_label:
            misclassified.append({
                'tweet': tweet,
                'true_label': true_label_str,
                'pred_label': pred_label_str
            })
        else:
            correct.append({
                'tweet': tweet,
                'true_label': true_label_str,
                'pred_label': pred_label_str
            })

    # Print some misclassified examples
    print("\nMisclassified Examples:")
    for i, example in enumerate(misclassified[:10]):  # Print first 10 misclassified examples
        print(f"Example {i + 1}:")
        print(f"Tweet: {' '.join(example['tweet'])}")
        print(f"True Label: {example['true_label']}")
        print(f"Predicted Label: {example['pred_label']}")
        print('-' * 50)

    # Print some correctly classified examples
    print("\nCorrectly Classified Examples:")
    for i, example in enumerate(correct[:10]):  # Print first 10 correct examples
        print(f"Example {i + 1}:")
        print(f"Tweet: {' '.join(example['tweet'])}")
        print(f"True Label: {example['true_label']}")
        print(f"Predicted Label: {example['pred_label']}")
        print('-' * 50)

def train_and_evaluate(model, lexicon, train_filepath, evaluation_filepath):
    # load our dataset
    print( "star to load training set")
    train_set = load_datafile(train_filepath)
    print( "Loaded training set")
    evaluation_set = load_datafile(evaluation_filepath)
    print( "Loaded Eval set")
    print('-'*50)

    # extract our features
    X_train, Y_train, X_test, Y_test = extract_features(model, lexicon, train_set, evaluation_set)
    Y_pred = []
    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'kernel': ['linear' ], #, 'rbf', 'poly'],
        'C': uniform(0.000001, 35) #,  # Continuous distribution for C
        #'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),  # Log-scale gamma values
        #'class_weight': [None, 'balanced'],
        #'degree': randint(2, 5),  # Random integer for degree
        #'coef0': uniform(0.0, 10.0)  # Continuous distribution for coef0
    }
    
    print( 'starting to train')    
    # # # Initialize the SVC model
    svc = SVC(C=0.4740852861788649, kernel='linear', probability=True, class_weight='balanced')  # Enable probability estimates

    #Used following to utilize RandomizedSearchCV
    #svc = SVC(  probability=True, class_weight='balanced')  # Enable probability estimates
    
    # # # Used RandomizedSearchCV to find the best hyperparameters
    # random_search = RandomizedSearchCV(
    #     estimator=svc,
    #     param_distributions=param_dist,
    #     n_iter=10, #50,  # Number of random combinations to try
    #     scoring='f1_weighted',
    #     cv=4,
    #     n_jobs=-1,
    #     random_state=42, 
    #     verbose=2
    # )
    # random_search.fit(X_train, Y_train)
    clf= svc.fit(X_train, Y_train)
    # print( 'DONE with train') 
    # # Get the best model
    # best_clf = random_search.best_estimator_
    
    # # Generate predictions using the best model
    # Y_pred = best_clf.predict(X_test)
    Y_pred = clf.predict(X_test)
    print( 'Done with predictions') 
    # # Generate classification report
    classification_report = metrics.classification_report(
        Y_test, Y_pred, digits=4, labels=[0, 1, 2], target_names=['negative', 'neutral', 'positive'])
    
    # # Print the best hyperparameters
    # print("Best Hyperparameters:", random_search.best_params_)

    analyze_predictions(evaluation_set, Y_test, Y_pred)

    return Y_pred, classification_report



# & "D:/Program Files/python3/python.exe" c:/Users/tezgi/Documents/Columbia/NLP-Winter-2025/HW1/sentiment-analysis-of-tweets/Solution-2.py --model Ngram --lexicon Hashtag --train ./data/train.csv --evaluation ./data/dev.csv 
#  & "D:/Program Files/python3/python.exe" c:/Users/tezgi/Documents/Columbia/NLP-Winter-2025/HW1/sentiment-analysis-of-tweets/Solution-2.py --model Ngram --lexicon Sentiment140  --train ./data/train.csv --evaluation ./data/dev.csv 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=['Ngram','Ling', 'Lex', 'Ngram+Lex', 'Ngram+Lex+Ling', 'Ngram+Ling', 'Ngram+custom','custom', 'Ngram+Lex+Ling+custom'],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon', dest='lexicon', required=True,
                        choices=['Hashtag', 'Sentiment140'])
    parser.add_argument('--train', dest='train_filepath', required=True,
                        help='Full path to the training file')
    parser.add_argument('--evaluation', dest='evaluation_filepath', required=True,
                        help='Full path to the evaluation file')
    args = parser.parse_args()

    predictions, classification_report = train_and_evaluate(
        args.model, args.lexicon, args.train_filepath, args.evaluation_filepath)
    
    print("Classification report:")
    print(classification_report)

