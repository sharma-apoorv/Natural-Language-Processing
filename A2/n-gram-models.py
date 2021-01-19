''' Assignment 2 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import re

from collections import Counter
from itertools import dropwhile

START_TOKEN = "<s>"
STOP_TOKEN = "</s>"
UNK = None
UNK_THRESHOLD = 3 #only values >= this will be retained as values. Other will be converted to UNK tokens

class FileParser:
    def __init__(self):
        self.TRAIN_FILE = "1b_benchmark.train.tokens"
        self.DEV_FILE   = "1b_benchmark.dev.tokens"
        self.TEST_FILE  = "1b_benchmark.test.tokens"
    
    def get_train_file_tokens(self):
        return self._tokenize(self._get_sentences(self.TRAIN_FILE))
    
    def get_dev_file_tokens(self):
        return self._tokenize(self._get_sentences(self.DEV_FILE))
    
    def get_test_file_tokens(self):
        return self._tokenize(self._get_sentences(self.TEST_FILE))

    def _flatten(self, l):
        return [word for sublist in l for word in sublist]

    def _tokenize(self, sentence_list):
        tokenized_sentences = [re.split("\s+", sentence.strip()) for sentence in sentence_list]
        return self._flatten(tokenized_sentences)

    def _get_sentences(self, file_path):
        
        l = []
        with open(file_path, "r") as f:
            l = f.readlines()
        
        # Add the start and stop tokens to each sentence in the file
        sentence_list = []
        for sentence in l:
            sentence_list.append(START_TOKEN + " " + sentence + " " + STOP_TOKEN)
        
        return sentence_list

class UnigramLanguageModel:
    def __init__(self, tokens, smoothing=False):
        self.token_freqs = Counter(tokens)
        self.corpus_length = 0
        self.unique_words = 0
        
        # Convert the keys to UNK tokens based on threshold value
        unk_freq_count = 0
        for key, count in dropwhile(lambda key_count: key_count[1] >= UNK_THRESHOLD, self.token_freqs.most_common()):
            unk_freq_count += self.token_freqs[key]
            del self.token_freqs[key]
        self.token_freqs[UNK] = unk_freq_count

        self.corpus_length = sum(self.token_freqs.values()) - self.token_freqs[START_TOKEN] - self.token_freqs[STOP_TOKEN]
        self.unique_words = len(list(self.token_freqs.keys())) - 1 #don't include START_TOKEN in this

        print(self.unique_words)

if __name__ == "__main__":
    fp = FileParser()
    train_tokens = fp.get_train_file_tokens()

    ulm = UnigramLanguageModel(train_tokens)