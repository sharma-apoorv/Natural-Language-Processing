''' Assignment 4 - NLP

Author: Apoorv Sharma
Description:
'''

import os
import sys
from collections import defaultdict as dd
from pprint import pprint as pp

from tqdm import tqdm
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

START = '<start>'
MASK = '<mask>'
SPACE = '<s>'
EOS = '<eos>'

class BigramModel:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print(f"Path: {file_path} does not exist")
            return None
        
        with open(file_path, "r") as f:
            bigrams = f.readlines()
        
        self.blm = dd(dict)

        for bigram in bigrams:
            bigram, prob = bigram.split('\t')[0], bigram.split('\t')[1]
            
            w1, w2 = bigram.split(' ')[0], bigram.split(' ')[1]
            prob = float(prob.strip())

            self.blm[w1][w2] = prob
    
        self._create_labels_dict()

    def _create_labels_dict(self):
        labels = list(self.blm.keys())
        labels.sort()
        labels.append(labels.pop(labels.index(EOS))) # add EOS to end
        labels = [labels.pop(labels.index(START))] + labels # add START to beginning

        self.label_to_idx = {k: v for v, k in enumerate(labels)}
        self.idx_to_label = {v: k for v, k in enumerate(labels)}

    def get_labels_to_index(self):
        return self.label_to_idx
    
    def get_index_to_labels(self):
        return self.idx_to_label
            
    def get_bigram_prob(self, w1, w2) -> float:
        if w1 in self.blm.keys() and w2 in self.blm[w1].keys():
            return self.blm[w1][w2] # p(w2 | w1) = prob
        return 0
    
    def get_max_w2(self, w1) -> str:
        return max(self.blm[w1], key=self.blm[w1].get)

class Viterbi:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print(f"Path: {file_path} does not exist")
            return None
        
        with open(file_path, "r") as f:
            l = f.readlines()
        
        self.sentences = []
        for sentence in l:
            sentence = sentence.split()
            self.sentences.append(sentence)
    
    def bi_gram_prob(self, blm: BigramModel):
        guessed_complete_sentences = []
        for sentence in self.sentences:
            guessed_sentence = sentence[:]
            N = len(guessed_sentence)

            for i in range(1, N):
                w1, w2 = guessed_sentence[i-1], guessed_sentence[i]

                if w2 == MASK:
                    w2 = blm.get_max_w2(w1)
                
                guessed_sentence[i-1], guessed_sentence[i] = w1, w2
            
            guessed_complete_sentences.append(guessed_sentence)
        
        return guessed_complete_sentences
    
    def get_transition_probability(self, labels, sentence_chars):
        R, C = len(labels), len(sentence_chars)
        tp = np.ones((R, C))
        tp = tp / np.sum(tp, axis=0)

        return tp
    
    def get_emission_probability(self, blm, labels, sentence_chars):
        R, C = len(labels), len(sentence_chars)
        ep = np.zeros((R, C))

        for j in range(1, len(sentence_chars)):
            w1, w2 = sentence_chars[j-1], sentence_chars[j]
            
            # if w1 == MASK:
            #     continue
            
            for label, i in labels.items():
                ep[i][j] = blm.get_bigram_prob(label, w1)

                if label == w2: #??: DO WE NEED THIS ?
                    ep[i][j] = 1
            
            w2 = blm.get_index_to_labels()[np.argmax(ep, axis=0)[j]]
            sentence_chars[j-1], sentence_chars[j] = w1, w2

        return ep

    def viterbi_algorithm(self, blm: BigramModel):
        labels = blm.get_labels_to_index()

        guessed_complete_sentences = []
        for sentence in tqdm(self.sentences, desc="Running Viterbi Algorithm"):
            guessed_sentence = sentence[:]
            N = len(guessed_sentence)

            tp = self.get_transition_probability(labels, guessed_sentence)
            ep = self.get_emission_probability(blm, labels, guessed_sentence)

            guessed_complete_sentences.append(guessed_sentence)
        
        return guessed_complete_sentences
    
    def write_sentences_to_file(self, sentence_list, file_path):
        sentence_list_strings = []
        
        for sentence in sentence_list:
            s = ' '.join(sentence)
            sentence_list_strings.append(s)
        
        s = '\n'.join(sentence_list_strings)

        with open(file_path, "w") as f:
            f.write(s)


if __name__ == "__main__":
    print("NLP - A4")

    blm = BigramModel('./lm.txt')

    v = Viterbi('./15pctmasked.txt')

    # complete_sentences = v.bi_gram_prob(blm)
    complete_sentences = v.viterbi_algorithm(blm)
    v.write_sentences_to_file(complete_sentences, './unmasked.txt')
