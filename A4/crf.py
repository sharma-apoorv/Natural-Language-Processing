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

        self.labels = labels
        self.label_to_idx = {k: v for v, k in enumerate(labels)}
        self.idx_to_label = {v: k for v, k in enumerate(labels)}

    def get_labels(self):
        return self.labels

    def get_labels_to_index(self):
        return self.label_to_idx
    
    def get_index_to_labels(self):
        return self.idx_to_label
            
    def get_bigram_prob(self, w1, w2) -> float:
        if w1 in self.blm.keys() and w2 in self.blm[w1].keys():
            return self.blm[w1][w2] # p(w2 | w1) = prob
        return 0
    
    def get_w2_given_w1(self, w1, w2, is_log_prob=True):
        if w1 in self.blm.keys() and w2 in self.blm[w1].keys():
            if is_log_prob: return np.log(self.blm[w1][w2]) # p(w2 | w1) = ln(prob)
            else: return self.blm[w1][w2] # p(w2 | w1) = ln(prob)

        if is_log_prob: return float('-inf')
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
    
    def viterbi_algorithm_2(self, blm: BigramModel):
        labels = blm.get_labels_to_index()

        tp_l = blm.get_labels_to_index()
        tp_i = blm.get_index_to_labels()
        
        guessed_complete_sentences = []
        for sentence in tqdm(self.sentences, desc="Running Viterbi Algorithm 2")[5:6]:
            guessed_sentence = sentence[:]

            R, C = len(labels), len(guessed_sentence)
            dp = np.zeros((R, C)) #init dp array to store calculations

            for i in range(R):
                dp[i][0] = 1 # All START columns with be initilized to 1
            
            dp_temp = np.zeros((R, ))
            for j in range(1, C):
                w1, w2 = guessed_sentence[j-1], guessed_sentence[j]
                prob = blm.get_bigram_prob(w1, w2)
                
                # For each label, calculate the dp probabilities
                for i, label in enumerate(labels):
                    dp_temp[tp_l[label]] = prob * dp[i][j-1]
                
                max_idx = np.argmax(dp_temp)
                guessed_sentence[j] = tp_i[max_idx]
                dp[i][j] = dp_temp[max_idx]
            
    def compute_missing_characters(self, blm: BigramModel):
        states = blm.get_labels()

        for sentence in tqdm(self.sentences[0:1], desc="Running Viterbi Algorithm 2"):
            # best_path = self.viterbi_wikipedia(sentence, states, blm)
            best_path = self.viterbi_michelle(sentence, states, blm)
            print(best_path)

    def viterbi_wikipedia(self, observation, states, blm: BigramModel):
        sentence = observation[:]

        R, C = len(states), len(sentence)

        # To hold p. of each state given each sentence.
        trellis = np.zeros((R, C))

        # Determine each hidden state's p. at time 0
        for i in range(R):
            trellis[i][0] = 1 # initial probability of START symbol
        
        # and now, assuming each state's most likely prior state, k
        for j in range(1, C):
            for i in range(R):
                w1, w2 = sentence[j-1], states[i]

                '''
                For each node in the prev column, the state that yields the max
                probability

                Index of this state is stored in k
                '''
                p_list = np.zeros((R,))
                for k in range(R):
                    transition_ps_to_w2 = blm.get_bigram_prob(w2, states[k]) # p(states[k] | w2)
                    print(f"{w2} | {states[k]} = {transition_ps_to_w2}")
                    p_list[k] = transition_ps_to_w2
                k = np.argmax(p_list) #the max INDEX that has the largest probability
                print(k,  states[k])
            
                break
            break


        #         trellis[i][j] = trellis[k][j-1] * blm.get_bigram_prob(states[k], w2) #fill in the largest p in trellis
        #         print(w1, w2, k, states[k])
            
        # np.savetxt('test.out', trellis)
        #     # max_char = states[np.argmax(trellis[:, j])]
        #     # sentence[j] = max_char

        # sentence_path = []
        # for j in range(C):
        #     k = np.argmax(trellis[:, j]) #the row INDEX with largest probability
        #     sentence_path.append(states[k])
        
        # return sentence_path

    def viterbi_michelle(self, observation, states, blm: BigramModel):

        sentence = observation[:]

        R, C = len(states), len(sentence)

        # To hold p. of each state given each sentence.
        trellis = np.full((R, C), -np.inf)
        
        # to hold the back pointers for cell
        back_pointer = np.zeros((R, C), dtype='int32')

        # Determine each hidden state's p. at time 0
        for i in range(R-1):
            if states[i] == START:
                trellis[i][0] = blm.get_w2_given_w1(START, START) # initial probability of START symbol
        
        # and now, assuming each state's most likely prior state, k
        for j in range(1, C-1):
            w1, w2 = sentence[j-1], sentence[j]

            for i in range(R-1):
                label = states[i]
                
                # Case 1: w1 and w2 are both known characters
                if w1 != MASK and w2 != MASK:
                    if w2 == label: trellis[i][j] = blm.get_w2_given_w1(w1, w2)
                    back_pointer[i][j] = np.argmax(trellis[:, j-1])
                
                # Case 2: curr is MASK and prev column is known
                elif w1 != MASK and w2 == MASK:
                    trellis[i][j] = blm.get_w2_given_w1(w1=w1, w2=label)
                    back_pointer[i][j] = np.argmax(trellis[:, j-1])

                # Case 3: curr is known and prev column is MASK
                elif w2 != MASK and w1 == MASK:
                    if w2 == label: trellis[i][j] = blm.get_w2_given_w1(w1=label, w2=w2)
                    back_pointer[i][j] = np.argmax(trellis[:, j-1])
                
                # Case 4: w1 and w2 are both MASK characters
                else:
                    t1 = np.full((R-1, ), -np.inf)

                    '''
                    Since the prev column is also a mask, we pick the current label and iterate through
                    all the other lables as well. Then find the max
                    '''
                    for k in range(R-1):
                        prev_label = states[k]
                        t1[k] = blm.get_w2_given_w1(w1=prev_label, w2=label)                    
                    trellis[i][j] = max(t1)
                    back_pointer[i][j] = np.argmax(t1)

        # Fill in the prob for <eos>
        trellis[R-1][C-1] = blm.get_w2_given_w1(w1=EOS, w2=EOS)
        back_pointer[R-1][C-1] = np.argmax(trellis[:, j-1])

        np.savetxt('back_pointer.out', back_pointer)

        # get the back pointers 
        guessed_sentence = [EOS]
        label_idx = back_pointer[R-1][C-1]
        for j in reversed(range(C-1)):
            guessed_sentence.append(states[label_idx])
            label_idx = back_pointer[label_idx][j]
        
        return list(reversed(guessed_sentence))

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

    # blm = BigramModel('./lm_avs.txt')
    blm = BigramModel('./lm.txt')

    # v = Viterbi('./15pctmasked_avs.txt')
    v = Viterbi('./15pctmasked.txt')

    # complete_sentences = v.bi_gram_prob(blm)
    # complete_sentences = v.viterbi_algorithm(blm)
    v.compute_missing_characters(blm)
    # v.write_sentences_to_file(complete_sentences, './unmasked.txt')
