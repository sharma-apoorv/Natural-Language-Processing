''' Assignment 4 - NLP

Author: Apoorv Sharma
Description:
'''

import argparse
import os
import operator
from collections import defaultdict as dd

import numpy as np
from tqdm import tqdm

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
    
    def get_max_from_key(self, w1) -> str:
        return max(self.blm[w1].items(), key=operator.itemgetter(1))

class Viterbi:
    def __init__(self, input_file_path, output_file_path):
        if not os.path.exists(input_file_path):
            print(f"Path: {input_file_path} does not exist")
            return None
        
        # Read and parse the input file
        with open(input_file_path, "r") as f:
            l = f.readlines()
        
        self.masked_sentences = []
        for sentence in l:
            sentence = sentence.split()
            self.masked_sentences.append(sentence)

        self.output_file_path = output_file_path

    def compute_missing_characters(self, blm: BigramModel):
        states = blm.get_labels()
        complete_sentences = []

        for sentence in tqdm(self.masked_sentences, desc="Running Viterbi Algorithm"):
            best_path = self.viterbi_algorithm(sentence, states, blm)
            complete_sentences.append(best_path)
        
        return complete_sentences

    def viterbi_algorithm(self, observation, states, blm: BigramModel):

        sentence = observation[:]

        idx_to_state = blm.get_index_to_labels()
        state_to_idx = blm.get_labels_to_index()
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
                    label_idx = state_to_idx[w2]
                    max_prev_trellis_value = max(trellis[:, j-1])
                    max_prev_trellis_label_idx = np.argmax(trellis[:, j-1])

                    trellis[label_idx][j] = max_prev_trellis_value + blm.get_w2_given_w1(w1=idx_to_state[max_prev_trellis_label_idx], w2=w2)
                    back_pointer[label_idx][j] = max_prev_trellis_label_idx
                    break
                
                # Case 2: curr is MASK and prev column is known
                elif w1 != MASK and w2 == MASK:
                    
                    max_prev_trellis_value = max(trellis[:, j-1])
                    max_prev_trellis_label_idx = np.argmax(trellis[:, j-1])

                    trellis[i][j] = max_prev_trellis_value + blm.get_w2_given_w1(w1=idx_to_state[max_prev_trellis_label_idx], w2=label)
                    back_pointer[i][j] = max_prev_trellis_label_idx

                # Case 3: curr is known and prev column is MASK
                elif w2 != MASK and w1 == MASK:
                    label_idx = state_to_idx[w2]

                    t1 = np.full((R-1, ), -np.inf)
                    for k in range(R-1):
                        prev_label = states[k]
                        t1[k] = blm.get_w2_given_w1(w1=prev_label, w2=w2) + trellis[k][j-1]
                    
                    trellis[label_idx][j] = max(t1)
                    back_pointer[label_idx][j] = np.argmax(t1)
                    break
                
                # Case 4: w1 and w2 are both MASK characters
                else:
                    t1 = np.full((R-1, ), -np.inf)

                    '''
                    Since the prev column is also a mask, we pick the current label and iterate through
                    all the other lables as well. Then find the max
                    '''
                    for k in range(R-1):
                        prev_label = states[k]
                        t1[k] = blm.get_w2_given_w1(w1=prev_label, w2=label) + trellis[k][j-1]
                    trellis[i][j] = max(t1)
                    back_pointer[i][j] = np.argmax(t1)
            
        # Fill in the prob for <eos>
        trellis[R-1][C-1] = blm.get_w2_given_w1(w1=EOS, w2=EOS)
        back_pointer[R-1][C-1] = np.argmax(trellis[:, C-2])

        np.savetxt('back_pointer.out', np.vstack((['header'] + sentence, np.column_stack((states, back_pointer.round(decimals=0))))), fmt="%-12s")
        np.savetxt('trellis.out', np.vstack((['header'] + sentence, np.column_stack((states, trellis.round(decimals=4))))), fmt="%-12s")

        # get the back pointers 
        guessed_sentence = [EOS]
        label_idx = back_pointer[R-1][C-1]
        for j in reversed(range(C-1)):
            guessed_sentence.append(states[label_idx])
            label_idx = back_pointer[label_idx][j]
        
        return list(reversed(guessed_sentence))

    def write_sentences_to_file(self, sentence_list):
        sentence_list_strings = []
        
        for sentence in sentence_list:
            s = ' '.join(sentence)
            sentence_list_strings.append(s)
        
        s = '\n'.join(sentence_list_strings)

        with open(self.output_file_path, "w") as f:
            f.write(s)

def sanity_check_output(masked_sentences, un_masked_sentences):
    print("Performing sanity check on output")

    # Ensure the first sentence matches the correct output:
    correct_out = ['<start>', 'I', '<s>', 'p', 'e', '<s>', 'm', 'a', 'n', 't', 'a', 't', 'i', 'o', 'n', '<s>', 'o', 'f', '<s>', 'G', 'e', 'o', 'r', 'g', 'i', 'a', "'", "'", '<s>', 'a', 'u', 'r', 'o', 'm', 'o', 'b', 'i', 'l', 'e', '<s>', 't', 'i', 't', 'l', 'e', '<s>', 'l', 'a', 'w', '<s>', 'w', 'a', 's', '<s>', 'a', 'l', '<s>', ',', '<s>', 'h', 'e', 'c', 'o', 'm', 'm', 'e', 'n', 'd', 'e', 'd', '<s>', 'b', 'e', '<s>', 't', 'h', 'e', '<s>', 'o', 'u', 't', 'g', 'o', 'i', 'n', 'g', '<s>', 'j', 'u', 'r', 'y', '<s>', '.', '<eos>']
    for correct_char, unmasked_char in zip(correct_out, un_masked_sentences[0]):
        if correct_char != unmasked_char:
            print(f'Error! Viterbi Algorithm is incorrect for sentence 1 Correct: {correct_char} : Unmasked {unmasked_char}')
            return

    for masked_sentence, unmasked_sentence in zip(masked_sentences, un_masked_sentences):

        # Ensure the length of 2 sentence is the same (same number of chars)
        lm, lum = len(masked_sentence), len(unmasked_sentence)
        if lm != lum:
            print(f"Error! The length of the sentences do not match")
            print(f"Masked Sentence: {masked_sentence}")
            print(f"Unmasked Sentence: {unmasked_sentence}")
        
        # Ensure we only changed the <mask> characters
        for i in range(lm):
            c_m, c_um = masked_sentence[i], unmasked_sentence[i]

            if c_m != MASK and (c_m != c_um):
                print(f"Error! Changed a known character")
                print(f"Changed {c_m} -> {c_um} at index: {i}")
            
            elif c_m == MASK and c_um == START:
                print(f"Changed a masked char to <start> token!")
                print(masked_sentence)
                print(unmasked_sentence)
                print("")
            
def parse_output_file(output_file_path):
    un_masked_sentences = []

    # Read in output file if it exists
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            l = f.readlines()
        for sentence in l:
            sentence = sentence.split()
            un_masked_sentences.append(sentence)
    
    return un_masked_sentences

if __name__ == "__main__":
    print("NLP - A4")

    parser = argparse.ArgumentParser(description='Viterbi Algorithm')
    parser.add_argument("-lm", "--lang-model", dest="lang_model_path", type=str, default="./lm.txt", required=False, help='This is the path to the language model file')
    parser.add_argument("-ip", "--input-file", dest="input_file_path", type=str, default="./15pctmasked.txt", required=False, help='This is the path to the file that contains the masked sentences')
    parser.add_argument("-op", "--output-file", dest="output_file_path", type=str, default="./unmasked.txt", required=False, help='This is the path to file that will be output')
    parser.add_argument("-t", "--sanity-check", dest="perform_sanity_check", action='store_true', help='Flag to indicate whether to perform sanity checking our not')
    args = parser.parse_args()

    blm = BigramModel(args.lang_model_path)
    v = Viterbi(input_file_path=args.input_file_path, output_file_path=args.output_file_path)

    complete_sentences = v.compute_missing_characters(blm)
    v.write_sentences_to_file(complete_sentences)

    if args.perform_sanity_check:
        sanity_check_output(v.masked_sentences, parse_output_file(args.output_file_path))