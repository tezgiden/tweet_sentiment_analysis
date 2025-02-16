# tweet sentiment analysis
This project implements a Tweet Sentiment analysis tool using SVN(Support Vector Machine).
This code was part of the Columbia University AI Certificate Program, NLP course project.

I implemented part of the features mentioned in https://aclanthology.org/S13-2053.pdf.

#I implemented 4 models for this project:
#NGRAM#: I utilized TfidfVectorizer to train the Vectorizor in the dataset.
#Lexicon features#: I implemented following features for Lexicon analysis.
       1 - total count of tokens (unigrams + bigrams) in the tweet with positive score > 0
       2 - total count of tokens (unigrams + bigrams) in the tweet with negative score > 0
       3 - summed positive score of all the unigrams and bigrams in the tweet 
       4 - summed negative score of all the unigrams and bigrams in the tweet
       5 - the max positive score of all the unigrams and bigrams in the tweet
       6 - the max negative score of all the unigrams and bigrams in the tweet
       7 - the max of the positive scores of the last unigram / bigram in the tweet (with score > 0)
       8 - the max of the negative scores of the last unigram / bigram in the tweet (with score > 0)  
#Linguistic features#:
       0 - number of tokens with all their characters capitalized
       1-23 - separate counts of each POS tag in the following sorted order:
               [
                   '!', '#', '$', '&', ',', '@', 'A', 'D', 'E', 'G', 'L', 'N', 
                   'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', '^', '~'
               ]
       24  - number of hashtags
       25  - number of words with one character repeated more than two times

#Custom feature (Negation)#: Analyzed the count of Negation sentences in the tweet

#Result#
In my case the f1-score accuracy scores were very close:
                 f1-score accuracy 
Ngram                 0.5238
Ngram+Lex             0.5488
Ngram+Lex+Ling        0.5417  
Ngram+Lex+Ling+custom 0.5476 

Best performing model was Ngram+Lex+Ling+custom
Classification report:
              precision    recall  f1-score   support

    negative     0.3503    0.4276    0.3851       145
     neutral     0.5806    0.6136    0.5967       352
    positive     0.6254    0.5306    0.5741       343

    accuracy                         0.5476       840
   macro avg     0.5188    0.5239    0.5186       840
weighted avg     0.5592    0.5476    0.5510       840
       
