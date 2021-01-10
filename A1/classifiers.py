''' Assignment 1 - NLP

Author: Apoorv Sharma
Description:

Q1.1: Sentiment lexicon-based classifier
Q1.2: Logistic regression classifier
Q1.3: Statistical significance (extra credit)
'''

# Import common required python files
import os
import random

import string
import re

import numpy as np

# Initialize data for sentiment analysis
FOLDER_NAME = 'txt_sentoken'
POS_DIR_NAME = 'pos'
NEG_DIR_NAME = 'neg'

POS_DIR_PATH = os.path.join(FOLDER_NAME, POS_DIR_NAME)
NEG_DIR_PATH = os.path.join(FOLDER_NAME, NEG_DIR_NAME)

TOTAL_TEST_FILES = 400
NUM_POS_TEST_FILES = random.randint(0, TOTAL_TEST_FILES)
NUM_NEG_TEST_FILES = TOTAL_TEST_FILES - NUM_POS_TEST_FILES

# Initialize sentiment lexicon
LEXICON_FOLDER_NAME = 'opinion-lexicon-English'
POS_LEXICON_FILE_NAME = 'positive-words.txt'
NEG_LEXICON_FILE_NAME = 'negative-words.txt'

POS_LEXICON_DIR_PATH = os.path.join(LEXICON_FOLDER_NAME, POS_LEXICON_FILE_NAME)
NEG_LEXICON_DIR_PATH = os.path.join(LEXICON_FOLDER_NAME, NEG_LEXICON_FILE_NAME)

def get_random_test_files() -> str:
    """ Function to randomly select TOTAL_TEST_FILES from the
    POS_DIR_PATH and NEG_DIR_PATH

    Parameters
    ----------
    None

    Returns
    -------
    A list of strings, where each string 
    contains the full path to the file
    """
    pos_files_list = [os.path.join(POS_DIR_PATH, file_name) for file_name in os.listdir(POS_DIR_PATH)]
    neg_files_list = [os.path.join(NEG_DIR_PATH, file_name) for file_name in os.listdir(NEG_DIR_PATH)]

    test_files_list = []
    random_files_index = random.sample(range(len(pos_files_list)), NUM_POS_TEST_FILES)
    test_pos_files_list = [pos_files_list[i] for i in random_files_index]

    random_files_index = random.sample(range(len(neg_files_list)), NUM_NEG_TEST_FILES)
    test_neg_files_list = [neg_files_list[i] for i in random_files_index]

    test_files_list = test_pos_files_list + test_neg_files_list
    random.shuffle(test_files_list)

    return test_files_list

def clean_lexicon_list_words(l: list) -> list:
    """ This function strips all characters in the list
    and removes any strings that begin with a ';'

    Parameters
    ----------
    l: A list of lexicon words

    Returns
    ----------
    A list of cleaned lexicon words.
    """

    clean_l = map(str.strip, l) #Remove trailing '\n' chars
    clean_l = [s for s in clean_l if not s.startswith(';')] #Remove initial explanation text

    return clean_l

def get_lexicon_words(file_path):
    """

    Parameters
    ----------

    Returns
    ----------
    """

    lex_words = []
    with open(file_path, encoding = "ISO-8859-1") as f:
        lex_words = f.readlines()
    lex_words = set(clean_lexicon_list_words(lex_words))

    return lex_words

def tokenize(text: str) -> list:
    """ Function to tokenize a sentence (break up into words)
    based on the TOKENIZER regex

    Parameters
    ----------
    text: A string that needs to be tokenized

    Returns
    ----------
    A list of tokens
    """
    TOKENIZER = re.compile(f'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~“”¨«»®´·º½¾¿¡§£₤‘’\n\t])')
    return TOKENIZER.sub(r' \1 ', text).split()

def is_digit_regex(s: str) -> bool:
    """Returns whether a string is a number
    or not

    Parameters
    ----------
    s: The string to check

    Returns
    ----------
    True: The string is a number
    False: The string is not a number
    """
    if re.match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True

def get_pos_neg_word_count(tokens: list, pos_lex_words: set, neg_lex_words: set) -> tuple:
    """ Function to count how many tokens are in the
    positive and negative lexicon word list passed in

    Parameters
    ----------
    tokens: List of tokens that need to be counted
    pos_lex_words: Set of positive lexicon words
    neg_lex_words: Set of negative lexicon words

    Returns
    ----------
    A tuple of positive and negative count
    """
    positive_words = 0
    negative_words = 0

    for token in tokens:
        if  token and len(token) > 1 and\
                (token not in string.punctuation) and\
                not is_digit_regex(token):\
            
            token = token.strip()

            if token in pos_lex_words: positive_words += 1
            elif token in neg_lex_words: negative_words += 1
    
    return (positive_words, negative_words)

def analyze_sentiment(test_files_list: list, classification_dict: dict):
    """ This is the main function for building a
    Sentiment lexicon-based classifier.

    Parameters
    ----------
    test_files_list: list of paths (strings), where each path
    should be the relative path to the file

    Returns
    ----------
    The classification_scores and true_value
    """

    # Lexicon words used for sentiment analysis
    pos_lex_words = get_lexicon_words(POS_LEXICON_DIR_PATH)
    neg_lex_words = get_lexicon_words(NEG_LEXICON_DIR_PATH)

    classification_scores = []
    true_labels = []

    for file in test_files_list:
        
        # Read the file to analyze
        with open(file) as f:
            sentences = f.readlines()

        # tokenize the sentences in the file
        tokens = []
        for sentence in sentences:
            tokens += tokenize(sentence) # Do not want to remove duplicate words, so we have more data
        
        # Get number of positive and negative words found in the file
        positive_words, negative_words = get_pos_neg_word_count(tokens, pos_lex_words, neg_lex_words)
        
        # Keep an array of all the scores we have (negative, positive)
        classification_score = [negative_words, positive_words]
        classification_scores.append(classification_score)
        
        # Maintain the true answer (negative, positive)
        true_label = [0, 0]
        if file.split('/')[1] == 'pos': true_label[1] += 1
        else: true_label[0] += 1
        true_labels.append(true_label)

        # Print for submitting assignment
        if true_label[0]: #file is actually negative
            classification_dict['neg'][file.split('/')[2]] = 'neutral'
            if positive_words > negative_words: classification_dict['neg'][file.split('/')[2]] = 'positive'
            else: classification_dict['neg'][file.split('/')[2]] = 'negative'
        else:
            classification_dict['pos'][file.split('/')[2]] = 'neutral'
            if positive_words > negative_words: classification_dict['pos'][file.split('/')[2]] = 'positive'
            else: classification_dict['pos'][file.split('/')[2]] = 'negative'

    
    return np.array(classification_scores), np.array(true_labels)

def compute_confusion_matrix_values(classification_scores, true_values) -> tuple:
    """ This function computes the confusion matrix scores namely:
    TP, TN, FP, FN

    Reference: https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/

    Parameters
    ----------
    classification_scores: The classification score for 
    each parameter that is being classified

    true_labels: The true score for each parameter
    that is being classified

    Returns
    ----------
    TP, TN, FP, FN

    """
    pred_labels = np.argmax(classification_scores, axis=1)
    true_labels = np.argmax(true_values, axis=1)

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    
    return TP, TN, FP, FN

def compute_accuracy(classification_scores: np.array, true_labels: np.array):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels)
    return (TP + TN) / (TP + TN + FP + FN)

def compute_precision(classification_scores: np.array, true_labels: np.array):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels)
    return TP / (TP + FP)

def compute_recall(classification_scores: np.array, true_labels: np.array):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels)
    return TP / (TP + TN)

def compute_f1_score(classification_scores: np.array, true_labels: np.array):
    precision = compute_precision(classification_scores, true_labels)
    recall = compute_recall(classification_scores, true_labels)

    return 2 * ((precision * recall) / (precision + recall))

def update_word_count(word, word_type, word_freq_dict):
    
    if (word, word_type) not in word_freq_dict.keys():
        word_freq_dict[(word, word_type)] = 0
    word_freq_dict[(word, word_type)] += 1

def get_clean_tokenized_words(file):
    # Read the file to analyze
    with open(file) as f:
        sentences = f.readlines()

    # tokenize the sentences in the file
    tokens = []
    for sentence in sentences:
        tokens += tokenize(sentence) # Do not want to remove duplicate words, so we have more data

    clean_words = []
    for word in tokens:
        if  word and len(word) > 1 and\
                    (word not in string.punctuation) and\
                    not is_digit_regex(word):
                
                clean_words.append(word.strip())
    
    return clean_words

def extract_features(file, bow_word_frequency):
    processed_words = get_clean_tokenized_words(file)
    
    # feature array
    features = np.zeros((1,3))
    
    # bias term
    features[0,0] = 1
    
    for word in processed_words:
        features[0,1] = bow_word_frequency.get((word, 1), 0) # positive reviews
        features[0,2] = bow_word_frequency.get((word, 0), 0) # negative reviews
    
    return features

def sigmoid(t): 
    # calculate the sigmoid of z
    h = 1 / (1+ np.exp(-t))
    
    return h

def gradientDescent(x, y, theta, alpha, num_iters, c):
    # get the number of samples in the training
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # find linear regression equation value, X and theta
        z = np.dot(x, theta) #score_LR
        
        # get the sigmoid of z
        h = sigmoid(z) # p_LR(Y=+1 | x; theta)
        
        # c is L2 regularizer term
        J = (-1/m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))) + (c * np.sum(theta)))
        
        # update the weights theta
        theta = theta - (alpha / m) * np.dot((x.T), (h - y))
   
    # J = float(J)
    return J, theta

def predict_sentiment(file, theta, word_freq_dict):

    x = extract_features(file, word_freq_dict)

    # make the prediction for x with learned theta values
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred

def test_accuracy(test_x, test_y, word_freq_dict, theta):

    # predict for the test sample with the learned weights for logistics regression
    for file in test_x:
        predicted_prob = predict_sentiment(file, theta, word_freq_dict)
        print(file, predicted_prob)
    
    # # assign the probability threshold to class
    # predicted_labels = np.where(predicted_probs > 0.5, 1, 0)
    
    # # calculate the accuracy
    # print(f"Own implementation of logistic regression accuracy is {len(predicted_labels[predicted_labels == np.array(test_y).reshape(-1,1)]) / len(test_y)*100:.2f}")


    # y_hat = []
    # for file in test_x:
        
    #     y_pred = predict_sentiment(file, theta, freqs_dict)
    #     print(y_pred)
        
    #     if y_pred.all() > (0.5, 1, 0):
           
    #         y_hat.append(1)
    #     else:
            
    #         y_hat.append(0)
    # m=len(y_hat)
    # y_hat=np.array(y_hat)
    # y_hat=y_hat.reshape(m)
    # test_y=test_y.reshape(m)
    
    # c=y_hat==test_y
    # j=0
    # for i in c:
    #     if i==True:
    #         j=j+1
    # accuracy = j/m
    # return accuracy

def get_bag_of_words(file_list):
    word_freq_dict = {}
    for file in file_list:
        
        # Read the file to analyze
        with open(file) as f:
            sentences = f.readlines()

        # tokenize the sentences in the file
        tokens = []
        for sentence in sentences:
            tokens += tokenize(sentence) # Do not want to remove duplicate words, so we have more data
        
        #TODO: Need to remove stop words!!
        for token in tokens:
            if  token and len(token) > 1 and\
                (token not in string.punctuation) and\
                not is_digit_regex(token):
                
                token = token.strip()

                if file.split('/')[1] == 'pos':
                    update_word_count(token, 1, word_freq_dict)
                else:
                    update_word_count(token, 0, word_freq_dict)
    
    return word_freq_dict

def binary_logistic_classifier(test_files_list):

    TRAIN_SPLIT = 80 / 100
    TEST_SPLIT = 1 - TRAIN_SPLIT
    # TEST_SPLIT = 1/100

    # Note: Pos and Neg not in even proportions! 
    train_x = test_files_list[:int(len(test_files_list) * TRAIN_SPLIT)]
    test_x = test_files_list[:int(len(test_files_list) * TEST_SPLIT)]

    train_y = np.array([1 if file.split('/')[1] == 'pos' else 0 for file in train_x])
    test_y = np.array([1 if file.split('/')[1] == 'pos' else 0 for file in train_x])

    # Get a bag of words dictionary
    word_freq_dict = get_bag_of_words(test_files_list)
    
    # Start training 
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], word_freq_dict)
    
    # print(X)

    Y = train_y
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-7, 1000, 0)

    # # print(f"The cost after training is {J:.8f}.")
    # print(test_files_list[:4])
    # print(f"The resulting vector of weights is {[np.round(t, 8) for t in np.squeeze(theta)]}")
    
    accuracy = test_accuracy(test_x, test_y, word_freq_dict, theta)
    # print(accuracy)
    
        

if __name__ == "__main__":
    classification_dict = {'pos': {}, 'neg':{}}

    # Random list of files
    test_files_list = get_random_test_files()

    # Question 1.1: Sentiment lexicon-based classifier
    # print("Question 1.1: Sentiment lexicon-based classifier")
    # classification_scores, true_labels = analyze_sentiment(test_files_list, classification_dict)
    # accuracy = compute_accuracy(classification_scores, true_labels)
    # f1_score = compute_f1_score(classification_scores, true_labels)

    # print(f"\nClassification of positive reviews:")
    # for file, classification in classification_dict['pos'].items():
    #     print(f"File: {file}\tModel Classification: {classification}")
    # print(f"\nClassification of negative reviews:")
    # for file, classification in classification_dict['neg'].items():
    #     print(f"File: {file}\tModel Classification: {classification}")

    # print(f"\nAccuracy: {accuracy:.2f}\tF1 Score: {f1_score:.2f}")

    binary_logistic_classifier(test_files_list)
