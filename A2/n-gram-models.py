''' Assignment 2 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import re
import math
import io

from collections import Counter
from itertools import dropwhile
from enum import Enum

START_TOKEN = "<s>"
STOP_TOKEN = "</s>"
UNK = "<UNK>"
UNK_THRESHOLD = 3 #only values >= this will be retained as values. Other will be converted to UNK tokens

class FileParser:
    def __init__(self, train_file="1b_benchmark.train.tokens", dev_file="1b_benchmark.dev.tokens", test_file="1b_benchmark.test.tokens"):
        self.TRAIN_FILE = train_file
        self.DEV_FILE   = dev_file
        self.TEST_FILE  = test_file
    
    def get_train_file_tokens(self):
        return self._tokenize(self._get_sentences(self.TRAIN_FILE))
    
    def get_dev_file_tokens(self):
        return self._tokenize(self._get_sentences(self.DEV_FILE))
    
    def get_test_file_tokens(self):
        return self._tokenize(self._get_sentences(self.TEST_FILE))
    
    def get_dev_file_sentence_tokens(self):
        return self._tokenize(self._get_sentences(self.DEV_FILE), flatten=False)
    
    def get_test_file_sentence_tokens(self):
        return self._tokenize(self._get_sentences(self.TEST_FILE), flatten=False)

    def _flatten(self, l):
        return [word for sublist in l for word in sublist]

    def _tokenize(self, sentence_list, flatten=True):
        tokenized_sentences = [re.split("\s+", sentence.strip()) for sentence in sentence_list]

        if flatten:
            return self._flatten(tokenized_sentences)
        
        return tokenized_sentences

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
    def __init__(self, tokens, smoothing=False, debug=False, file_pointer=None):
        self.unigram_freqs = Counter(tokens)
        self.unigram_corpus_length = 0
        self.num_unique_unigrams = 0
        self.debug = debug
        self.file_pointer = file_pointer
        
        # Convert the keys to UNK tokens based on threshold value
        unk_freq_count = 0
        for key, count in dropwhile(lambda key_count: key_count[1] >= UNK_THRESHOLD, self.unigram_freqs.most_common()):
            unk_freq_count += self.unigram_freqs[key]
            del self.unigram_freqs[key]
        self.unigram_freqs[UNK] = unk_freq_count

        self.unigram_corpus_length = sum(self.unigram_freqs.values()) - self.unigram_freqs[START_TOKEN]
        self.num_unique_unigrams = len(list(self.unigram_freqs.keys())) - 1 #don't include START_TOKEN in this

        if self.debug:
            if isinstance(file_pointer, io.TextIOWrapper):
                file_pointer.write("\nUNIGRAM MODEL INIT\n")
                for k,v in  self.unigram_freqs.items():
                    file_pointer.write("{} {}\n".format(k,v))
            else:
                print("\nUNIGRAM MODEL INIT\n")
                print(self.unigram_freqs)
    
    def _get_unigram_corpus_count(self, tokens):
        words = Counter(tokens)
        return sum(list(words.values())) - words.get(START_TOKEN, 0)

    def _get_unigram_probability(self, word):
        prob_numerator = self.unigram_freqs.get(word, self.unigram_freqs[UNK])
        prob_denominator = self.unigram_corpus_length
        prob = float(prob_numerator) / float(prob_denominator)
        
        if self.debug:
            if isinstance(self.file_pointer, io.TextIOWrapper):
                self.file_pointer.write(f"Unigram Word: {word} Prob = {prob_numerator}/{prob_denominator} = {prob:.5e}\n")
            else:
                print(f"Unigram Word: {word} Prob = {prob_numerator}/{prob_denominator} = {prob:.5e}")

        return prob
    
    def get_unigram_sentence_probability(self, sentence, normalize=True):
        sentence_prob_sum = 0
        for word in sentence:
            if word == START_TOKEN or word == STOP_TOKEN: continue

            word_prob = self._get_unigram_probability(word)
            if word_prob: sentence_prob_sum += math.log(word_prob)
        
        return math.exp(sentence_prob_sum)
    
    def get_unigram_perplexity(self, tokens):
        unigram_perplexity = 0
        for token in tokens:
            unigram_probability = self._get_unigram_probability(token)
            unigram_perplexity += math.log(unigram_probability, 2)
        
        num_words = self._get_unigram_corpus_count(tokens)
        unigram_perplexity = (-1 / num_words) * unigram_perplexity

        return math.pow(2, unigram_perplexity)

class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, tokens, smoothing=False, debug=False, file_pointer=None):
        UnigramLanguageModel.__init__(self, tokens, smoothing, debug, file_pointer)

        self.bigram_corpus_length = 0
        self.num_unique_bigrams = 0

        self.bigram_freqs = {}
        for i in range(1, len(tokens)):
            t1, t2 = tokens[i-1], tokens[i]

            if t1 not in self.unigram_freqs: t1 = UNK
            if t2 not in self.unigram_freqs: t2 = UNK

            self.bigram_freqs[(t1, t2)] = self.bigram_freqs.get((t1, t2), 0) + 1
            
        self.bigram_corpus_length = sum(self.bigram_freqs.values())
        self.num_unique_bigrams = len(list(self.bigram_freqs.keys()))

        if self.debug:
            if isinstance(self.file_pointer, io.TextIOWrapper):
                self.file_pointer.write("\nBIGRAM MODEL INIT\n")
                for k,v in  self.bigram_freqs.items():
                    self.file_pointer.write("{} {}\n".format(k,v))
            else:
                print("\nBIGRAM MODEL INIT\n")
                print(self.bigram_freqs)
    
    def _get_bigram_probability(self, bigram):
        curr_word, next_word = bigram

        prob_numerator = self.bigram_freqs.get((curr_word, next_word), 0)
        prob_denominator = self.unigram_freqs.get(curr_word, self.unigram_freqs[UNK]) #TODO: Should this be a probability of UNK ?

        if prob_denominator == 0:
            return 0

        prob = float(prob_numerator) / float(prob_denominator)

        if self.debug:
            if isinstance(self.file_pointer, io.TextIOWrapper):
                self.file_pointer.write(f"Bigram: {bigram} Prob = {prob_numerator}/{prob_denominator} = {prob:.5e}\n")
            else:
                print(f"Bigram: {bigram} Prob = {prob_numerator}/{prob_denominator} = {prob:.5e}")

        return prob

    def get_bigram_sentence_probability(self, sentence, normalize=True):
        sentence_prob_sum = 0
        for i in range(1, len(sentence)):
            curr_word, next_word = sentence[i-1], sentence[i]

            bigram_prob = self._get_bigram_probability((curr_word, next_word))
            if bigram_prob: sentence_prob_sum += math.log(bigram_prob)
        
        return math.exp(sentence_prob_sum)
    
    def _get_bigram_corpus_count(self, tokens):
        num_bigrams = 0
        if tokens:
            num_bigrams = len(tokens) - 1 # There are n-1 bigrams in a list of n words

        return num_bigrams

    def get_bigram_perplexity(self, tokens):
        bigram_perplexity = 0
        for i in range(1, len(tokens)):
            curr_word, next_word = tokens[i-1], tokens[i]
            bigram_probability = self._get_bigram_probability((curr_word, next_word))
            
            try:
                bigram_perplexity += math.log(bigram_probability, 2)
            except:
                # TODO: What are we supposed to do here ??
                continue
        
        num_words = self._get_bigram_corpus_count(tokens)
        bigram_perplexity = (-1 / num_words) * bigram_perplexity

        return math.pow(2, bigram_perplexity)

if __name__ == "__main__":

    output_file_name = "output.txt"
    f = open(output_file_name, "w")

    from pprint import pprint as pp
    # fp = FileParser(train_file="1b_mini.test.tokens")
    fp = FileParser()
    train_tokens = fp.get_train_file_tokens()
    dev_tokens = fp.get_dev_file_tokens()
    test_tokens = fp.get_test_file_tokens()

    blm = BigramLanguageModel(train_tokens, debug=False, file_pointer=f)

    f.write(f"\n*********Model Information*********\n")
    f.write('UNIGRAM MODEL\n')
    f.write(f"Corpus Length: {blm.unigram_corpus_length}\tUnique Unigrams: {blm.num_unique_unigrams}\n\n")

    f.write('BIGRAM MODEL\n')
    f.write(f"Corpus Length: {blm.bigram_corpus_length}\tUnique Bigrams: {blm.num_unique_bigrams}\n")

    f.write(f"\n*********Model Evaluation*********\n")
    f.write("Perplexity Scores\n")
    f.write(f"Unigram Perplexity (train): {blm.get_unigram_perplexity(train_tokens):.4f}\n")
    f.write(f"Unigram Perplexity (dev): {blm.get_unigram_perplexity(dev_tokens):.4f}\n")
    f.write(f"Unigram Perplexity (test): {blm.get_unigram_perplexity(test_tokens):.4f}\n\n")

    f.write(f"Bigram Perplexity (train): {blm.get_bigram_perplexity(train_tokens):.4f}\n")
    f.write(f"Bigram Perplexity (dev): {blm.get_bigram_perplexity(dev_tokens):.4f}\n")
    f.write(f"Bigram Perplexity (test): {blm.get_bigram_perplexity(test_tokens):.4f}\n\n")

    f.close()