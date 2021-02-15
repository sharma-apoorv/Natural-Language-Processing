''' Assignment 4 - NLP

Author: Apoorv Sharma
Description:
'''

import os
from pprint import pprint as pp
from collections import defaultdict as dd

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
            
    def get_bigram_prob(self, w1, w2) -> float:
        if w1 in self.blm.keys() and w2 in self.blm[w1].keys():
            return self.blm[w1][w2] # p(w2 | w1) = prob
        return -1
    
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

    complete_sentences = v.bi_gram_prob(blm)
    v.write_sentences_to_file(complete_sentences, './unmasked.txt')