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

def analyze_sentiment():
    """ This is the main function for building a
    Sentiment lexicon-based classifier.

    Parameters
    ----------
    None

    Returns
    ----------
    The classification_scores and true_value
    """

    # Random list of files
    test_files_list = get_random_test_files()

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

if __name__ == "__main__":
    # Question 1.1: Sentiment lexicon-based classifier
    print("Question 1.1: Sentiment lexicon-based classifier")
    classification_scores, true_labels = analyze_sentiment()
    accuracy = compute_accuracy(classification_scores, true_labels)
    f1_score = compute_f1_score(classification_scores, true_labels)
    print(f"Accuracy:{accuracy:.2f}\tF1 Score:{f1_score:.2f}")
