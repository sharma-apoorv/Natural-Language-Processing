''' Assignment 3 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import random
import re
from collections import Counter

# Import special libraries
import numpy as np

import nltk
nltk.download('punkt')
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm

# Hyperparameters
MAX_SEQ_LEN = -1  # -1 for no truncation
UNK_THRESHOLD = 5
BATCH_SIZE = 128
N_EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_RNN_LAYERS = 2

# Data cleaning parms
PAD = "@@PAD@@"
UNK = "@@UNK@@"

class GloVeWordEmbeddings():
    def __init__(self, glove_file_path, num_dims):

        if not os.path.exists(glove_file_path):
            print("Error ... Not a valid glove path")
            return
        
        self.embeddings_dict = {}

        max_words = 10000
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()

                # create a dict of word -> positions
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                
                self.embeddings_dict[word] = vector

                # if i == max_words: break

    def _get_cosine_similarity(self, vecA: np.array, vecB: np.array):
        return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

    def _get_closest_words(self, embedding):
        return sorted(self.embeddings_dict.keys(), key=lambda w: self._get_cosine_similarity(self.embeddings_dict[w], embedding), reverse=True)
    
    def _get_embedding_for_word(self, word: str) -> np.array:
        if word in self.embeddings_dict.keys():
            return self.embeddings_dict[word]
        return np.array([])

    def get_x_closest_words(self, word, num_closest_words=1) -> list: 
        '''
        References:
        https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
        https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db
        '''

        embedding = self._get_embedding_for_word(word)
        if embedding.size == 0:
            print(f"{word} does not exist in the embeddings.")
            return []
        
        return self._get_closest_words(embedding)[1:num_closest_words+1]
    
    def get_word_analogy_closest_word(self, w1, w2, w3, num_closest_words=1):
        e1 = self._get_embedding_for_word(w1)
        e2 = self._get_embedding_for_word(w2)
        e3 = self._get_embedding_for_word(w3)

        if e1.size == 0 or e2.size == 0 or e3.size == 0:
            print(f"{w1}:{e1.size}  {w2}:{e2.size}  {w3}:{e3.size}")
            return []

        embedding = e1 - e2 + e3
        return self._get_closest_words(embedding)[1:num_closest_words+1]

class CornellMovieReviewFiles():
    def __init__(self, 
                folder_name='txt_sentoken', pos_dir_name = 'pos', neg_dir_name = 'neg',\
                train_percent = 60, dev_percent = 20\
    ):

        self.POS_DIR_PATH = os.path.join(folder_name, pos_dir_name)
        self.NEG_DIR_PATH = os.path.join(folder_name, neg_dir_name)

        # Combine all paths from positive and negative reviews
        pos_files_list = [os.path.join(self.POS_DIR_PATH, file_name) for file_name in os.listdir(self.POS_DIR_PATH)]
        neg_files_list = [os.path.join(self.NEG_DIR_PATH, file_name) for file_name in os.listdir(self.NEG_DIR_PATH)]
        
        movie_review_file_list = pos_files_list + neg_files_list
        random.shuffle(movie_review_file_list)

        # How many files will be there in each set ?
        NUM_FILES = len(movie_review_file_list)
        NUM_TRAIN_FILES = int(NUM_FILES * (train_percent / 100))
        NUM_DEV_FILES = int(NUM_FILES * (dev_percent / 100))
        NUM_TEST_FILES = NUM_FILES - (NUM_TRAIN_FILES + NUM_DEV_FILES)

        self.train_files_list, self.dev_files_list, self.test_files_list = [], [], []

        # Generate a list of test files
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_TRAIN_FILES)
        self.train_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        # Generate a list of dev files
        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_DEV_FILES)
        self.dev_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        self.test_files_list = movie_review_file_list[:]

    def _apply_vocab(self, data):
        """
        Applies the vocabulary to the data and maps the tokenized sentences to vocab indices as the
        model input.
        """
        for review in data:
            review[self.TEXT] = [self.token_to_index.get(token, self.token_to_index[UNK]) for token in review[self.TEXT]]

    def _create_vocab(self, data, unk_threshold=UNK_THRESHOLD):
        """
        Creates a vocabulary with tokens that have frequency above unk_threshold and assigns each token
        a unique index, including the special tokens.
        """
        counter = Counter(token for review in data for token in review[self.TEXT])
        vocab = {token for token in counter if counter[token] > unk_threshold}
        # print(f"Vocab size: {len(vocab) + 2}")  # add the special tokens
        # print(f"Most common tokens: {counter.most_common(10)}")
        token_to_idx = {PAD: 0, UNK: 1}
        for token in vocab:
            token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def _create_dataset(self, files_list):

        data = []
        for file in files_list:
            
            # Read the file to analyze
            with open(file) as f:
                label = -1
                if file.split("/")[1] == "pos": label = self.POS
                elif file.split("/")[1] == "neg": label = self.NEG

                data.append({
                    self.RAW: f.read(),
                    self.LABEL: label
                })
        
        return data
            
    def _tokenize(self, data, max_seq_len=MAX_SEQ_LEN):
        """
        Here we use nltk to tokenize data. There are many other possibilities. We also truncate the
        sequences so that the training time and memory is more manageable. You can think of truncation
        as making a decision only looking at the first X words.

        Reference: CSE-447 Section AB
        """
        for example in data:
            example[self.TEXT] = []
            for sent in nltk.sent_tokenize(example[self.RAW]):
                example[self.TEXT].extend(nltk.word_tokenize(sent))
            if max_seq_len >= 0:
                example[self.TEXT] = example[self.TEXT][:max_seq_len]

    def pre_process_data(self):
        # Create a dictionary with labels and raw text for each dataset
        self.train_dataset, self.dev_dataset, self.test_dataset = [], [], []
        self.RAW, self.LABEL, self.TEXT = "raw", "label", "text"
        self.POS, self.NEG = 1, 0

        # Create a list of information for file to parse
        self.train_dataset = self._create_dataset(self.train_files_list)
        self.dev_dataset = self._create_dataset(self.dev_files_list)
        self.test_dataset = self._create_dataset(self.test_files_list)

        # Tokenize the all the 'raw' sentences
        for data in (self.train_dataset, self.dev_dataset, self.test_dataset):
            self._tokenize(data)

        # Map the vocab to an index
        self.token_to_index = self._create_vocab(self.train_dataset)

        for data in (self.train_dataset, self.dev_dataset, self.test_dataset):
            self._apply_vocab(data)
    
    def get_token_to_index_mapping(self):
        return self.token_to_index
    
    def get_train_data(self):
        return self.train_dataset
    
    def get_dev_data(self):
        return self.dev_dataset
    
    def get_test_data(self):
        return self.test_dataset

def word_embedding_questions(glove, output_file):

    _num_closest_words = 1

    output_file.write(f"*********Question 3.1*********\n")

    find_closest_words_list = ["dog", "whale", "before", "however", "fabricate"]
    for word in find_closest_words_list:
        output_file.write(f"{word}: {glove.get_x_closest_words(word, num_closest_words=_num_closest_words)}\n")
    
    output_file.write(f"\n*********Question 3.2*********\n")

    word_analogy_list = [("dog", "puppy", "cat"), ("speak", "speaker", "sing"), ("france", "french", "england"), ("france", "wine", "england")]
    for w1, w2, w3 in word_analogy_list:
        output_file.write(f"{w1} : {w2} :: {w3} : {glove.get_word_analogy_closest_word(w1, w2, w3, num_closest_words=_num_closest_words)}\n")

def sentiment_classification(output_file):
    cmr = CornellMovieReviewFiles()
    cmr.pre_process_data()
    
    token_to_index_mapping = cmr.get_token_to_index_mapping()
    train_data, dev_data, test_data = cmr.get_train_data(), cmr.get_dev_data(), cmr.get_test_data()

if __name__ == '__main__':
    # glove = GloVeWordEmbeddings('glove.42B.300d.txt', 300)

    output_file_name = "output.txt"
    output_file = open(output_file_name, "w")

    # Question 3.1 and 3.2
    # word_embedding_questions(glove, output_file)

    # Question 3.3
    sentiment_classification(output_file_name)


    print(f"Done! Check {output_file_name} for details.")