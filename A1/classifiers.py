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

COMMON_STOP_WORDS = set(["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"])

NUM_FEATURES = 0

def get_test_train_files_split() -> str:
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
    test_pos_files_list = [pos_files_list[i] for i in random_files_index] # pos test files list
    train_pos_file_list = [pos_files_list[i] for i in range(len(pos_files_list)) if i not in random_files_index] # pos train file list

    random_files_index = random.sample(range(len(neg_files_list)), NUM_NEG_TEST_FILES)
    test_neg_files_list = [neg_files_list[i] for i in random_files_index]
    train_neg_file_list = [neg_files_list[i] for i in range(len(neg_files_list)) if i not in random_files_index] # pos train file list

    test_files_list = test_pos_files_list + test_neg_files_list
    train_files_list = train_pos_file_list + train_neg_file_list

    random.shuffle(test_files_list)
    random.shuffle(train_files_list)

    return test_files_list, train_files_list

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

def compute_confusion_matrix_values(classification_scores, true_values, take_max=True) -> tuple:
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
    pred_labels, true_labels = classification_scores, true_values
    if take_max:
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

def compute_accuracy(classification_scores: np.array, true_labels: np.array, take_max=True):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels, take_max)
    return ((TP + TN) / (TP + TN + FP + FN)) * 100

def compute_precision(classification_scores: np.array, true_labels: np.array, take_max=True):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels, take_max)
    return TP / (TP + FP)

def compute_recall(classification_scores: np.array, true_labels: np.array, take_max=True):
    TP, TN, FP, FN = compute_confusion_matrix_values(classification_scores, true_labels, take_max)
    return TP / (TP + TN)

def compute_f1_score(classification_scores: np.array, true_labels: np.array, take_max=True):
    precision = compute_precision(classification_scores, true_labels, take_max)
    recall = compute_recall(classification_scores, true_labels, take_max)

    return (2 * ((precision * recall) / (precision + recall))) * 100

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
                    not is_digit_regex(word) and\
                    word not in COMMON_STOP_WORDS:
                
                clean_words.append(word.strip())
    
    return clean_words

def extract_features(file, bow_word_frequency, tf_idf_features):
    global NUM_FEATURES
    word_features_len = tf_idf_features.shape[0]
    
    processed_words = get_clean_tokenized_words(file)
    
    # feature array
    features = np.zeros((1,NUM_FEATURES))
    # features = np.concatenate( (np.zeros((1,NUM_FEATURES-word_features_len)), tf_idf_features[:,None]), axis=1)
    # features = np.column_stack( (np.zeros((1,NUM_FEATURES-word_features_len)), tf_idf_features) )
    
    # bias term
    features[0,0] = 1
    
    for word in processed_words:
        features[0,1] += bow_word_frequency.get((word, 1), 0) # positive reviews
        features[0,2] += bow_word_frequency.get((word, 0), 0) # negative reviews
    
    # features[0,3] = np.log(len(processed_words)) # word count
    # features[0,4] = 1 if "no" in processed_words else 0 # if a negation is in the review

    return features

def sigmoid(x):
    # return np.where(x >= 0, 
    #                 1 / (1 + np.exp(-x)), 
    #                 np.exp(x) / (1 + np.exp(x)))
    
    return 1 / (1 + np.exp(-x))

def gradient_descent(x, y, theta, alpha, num_iters):
    # get the number of samples in the training
    m = x.shape[0]
    J = 0

    for i in range(num_iters):
        
        # find linear regression equation value, X and theta
        z = np.dot(x, theta) #score_LR
        
        # get the sigmoid of z
        h = sigmoid(z) # p_LR(Y=+1 | x; theta)
        
        # cross-entropy loss function
        # J = (-1/m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))))
        # if(i % 10 == 0):
        #     print(f"Loss after epoch {i} is: {float(J):.5f}")

        # update the weights theta
        theta = theta - (alpha / m) * np.dot((x.T), (h - y))

    J = float(J)
    return J, theta

def predict_sentiment(x, theta):
    # make the prediction for x with learned theta values
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred

def test_classifier(test_x, theta):

    # predict for the test sample with the learned weights for logistics regression
    predicted_probs = predict_sentiment(test_x, theta)

    # assign the probability threshold to class
    predicted_labels = np.where(predicted_probs > 0.5, 1, 0)

    return predicted_labels

def get_bag_of_words(file_list):
    word_freq_dict = {}
    DF_temp = {}
    for i, file in enumerate(file_list):
        tokens = get_clean_tokenized_words(file)
        
        for token in tokens:
            if file.split('/')[1] == 'pos':
                update_word_count(token, 1, word_freq_dict)
            else:
                update_word_count(token, 0, word_freq_dict)
            
            if token not in DF_temp.keys():
                DF_temp[token] = set()
            DF_temp[token].add(i)
    
    from collections import Counter

    tf_idf = {}
    N = len(file_list)

    for i, file in enumerate(file_list):
        tokens = get_clean_tokenized_words(file)
        counts = Counter(tokens)

        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counts[token] / words_count #tf(t,d) = count of t in d / number of words in d
            df = len(DF_temp[token]) #df(t) = occurrence of t in documents

            idf = np.log(N/(df+1))
            tf_idf[(i, token)] = tf*idf

    total_vocab = list(DF_temp.keys())
    total_vocab_size = len(total_vocab)

    D = np.zeros((N, total_vocab_size))
    for i in tf_idf:
        doc_index = i[0]
        token = i[1]

        index_of_word = total_vocab.index(token)
        D[doc_index][index_of_word] = tf_idf[i]

    return word_freq_dict, D, tf_idf, total_vocab

def get_D_test(tf_idf_train, tf_idf_test, total_vocab, test_files_list):
    N = len(test_files_list)
    total_vocab_size = len(total_vocab)

    D = np.zeros((N, total_vocab_size))
    for i in range(N):
        for j in range(total_vocab_size):
            token = total_vocab[j]
            
            D[i][j] = 0

            if (i, token) in tf_idf_test.keys():
                D[i][j] = tf_idf_test[(i, token)]
    
    return D

def binary_logistic_classifier(test_files_list, train_files_list, classification_dict):
    global NUM_FEATURES

    # Infer the true outputs based on the file names
    train_y = np.array([1 if file.split('/')[1] == 'pos' else 0 for file in train_files_list])
    test_y = np.array([1 if file.split('/')[1] == 'pos' else 0 for file in test_files_list])

    # Get a bag of words dictionary
    word_freq_dict, tf_idf_features, tf_idf_train, total_vocab = get_bag_of_words(train_files_list)
    
    # word_freq = get_word_frequency(train_files_list)
    # features = list(word_freq.keys())
    NUM_FEATURES = 3

    # Extract wanted features from the train inputs
    X = np.zeros((len(train_files_list), NUM_FEATURES))
    for i in range(len(train_files_list)):
        X[i, :] = extract_features(train_files_list[i], word_freq_dict, tf_idf_features[i])
    X = np.concatenate((X, tf_idf_features), axis=1)
    
    word_freq_dict_test, _, tf_idf_test, _ = get_bag_of_words(test_files_list)
    tf_idf_features_test = get_D_test(tf_idf_train, tf_idf_test, total_vocab, test_files_list)

    # Extract wanted features from the test inputs
    test_x = np.zeros((len(test_files_list), NUM_FEATURES))
    for index, file in enumerate(test_files_list):
        test_x[index, :] = extract_features(file, word_freq_dict_test, tf_idf_features_test[index])
    test_x = np.concatenate((test_x, tf_idf_features_test), axis=1)

    NUM_FEATURES += tf_idf_features.shape[1]
    Y = np.array(train_y).reshape(-1,1)
    J, theta = gradient_descent(X, Y, np.zeros((NUM_FEATURES, 1)), 1e-4, 1000)

    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[np.round(t, 8) for t in np.squeeze(theta)]}")
    
    predicted_labels = test_classifier(test_x, theta)
    predicted_labels = predicted_labels.reshape((1,-1))[0]

    for i, file in enumerate(test_files_list):
        # Print for submitting assignment
        if test_y[i]: #file is actually positive
            classification_dict['pos'][file.split('/')[2]] = 'neutral'
            if predicted_labels[i]: classification_dict['pos'][file.split('/')[2]] = 'positive'
            else: classification_dict['pos'][file.split('/')[2]] = 'negative'
        else:
            classification_dict['neg'][file.split('/')[2]] = 'neutral'
            if predicted_labels[i]: classification_dict['neg'][file.split('/')[2]] = 'positive'
            else: classification_dict['neg'][file.split('/')[2]] = 'negative'

    return predicted_labels, test_y

if __name__ == "__main__":

    output_file_name = "output.txt"
    f = open(output_file_name, "w")

    classification_dict = {'pos': {}, 'neg':{}}

    # Random list of files
    test_files_list, train_files_list = get_test_train_files_split()

    # Question 1.1: Sentiment lexicon-based classifier
    f.write("Question 1.1: Sentiment lexicon-based classifier\n")
    classification_scores, true_labels = analyze_sentiment(test_files_list, classification_dict)
    accuracy = compute_accuracy(classification_scores, true_labels)
    f1_score = compute_f1_score(classification_scores, true_labels)

    f.write(f"\nClassification of positive reviews:\n")
    for file, classification in classification_dict['pos'].items():
        f.write(f"File: {file}\tModel Classification: {classification}\n")
    f.write(f"\nClassification of negative reviews:\n")
    for file, classification in classification_dict['neg'].items():
        f.write(f"File: {file}\tModel Classification: {classification}\n")

    f.write(f"\nAccuracy: {accuracy:.2f}\tF1 Score: {f1_score:.2f}\n")

    # Question 1.2: Logistic regression classifier
    logistic_classification_dict = {'pos': {}, 'neg':{}}
    f.write("\nQuestion 1.2: Logistic regression classifier\n")
    classification_scores, true_labels = binary_logistic_classifier(test_files_list, train_files_list, logistic_classification_dict)
    accuracy = compute_accuracy(classification_scores, true_labels, False)
    f1_score = compute_f1_score(classification_scores, true_labels, False)

    f.write(f"\nClassification of positive reviews:\n")
    for file, classification in logistic_classification_dict['pos'].items():
        f.write(f"File: {file}\tModel Classification: {classification}\n")
    f.write(f"\nClassification of negative reviews:\n")
    for file, classification in logistic_classification_dict['neg'].items():
        f.write(f"File: {file}\tModel Classification: {classification}\n")

    f.write(f"\nAccuracy: {accuracy:.2f}\tF1 Score: {f1_score:.2f}\n")
    
    f.close()
    print(f"Done! Check {output_file_name} for results!")