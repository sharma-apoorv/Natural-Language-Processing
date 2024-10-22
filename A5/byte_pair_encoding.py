import json
import os
import re
import string
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from pprint import pprint as pp

plt.style.use('seaborn-whitegrid')

SPACE_SYM = '<s>'

# from bpe import Encoder

class BytePairEncoding:
    def __init__(   self, 
                    file_path,
                    space_symbol = SPACE_SYM
                ) -> None:
        
        self.space = space_symbol

        if not os.path.exists(file_path):
            print(f"Error! {file_path} does not exist!")
            return None
        
        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Step 0: Clean text
        sentences_list = self._clean_sentences(lines)

        # Step 1: Append, to each word, a special <s> symbol marking the end of a word
        self.vocab, space_marked_sentences_list = self._add_word_space_symbols(sentences_list)
        self.types_vocab = set()
    

    def _clean_sentences(self, sentences_list):
        # sentences_list = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences_list]
        return list(map(str.strip, sentences_list))
    

    def _add_word_space_symbols(self, sentences_list):
        modified_sentences_list = []

        for sentence in sentences_list:
            word_list = sentence.split(" ")
            tokens = [" ".join(word) + ' ' + self.space for word in word_list]
            # tokens = " ".join(word_list) + ' ' + self.space
            modified_sentences_list.append(tokens)
        
        vocab = Counter(self._flatten(modified_sentences_list))
        # vocab = Counter(modified_sentences_list)
            
        return vocab, modified_sentences_list


    def _get_pair_counts(self):
        pairs = defaultdict(int)
        for word, frequency in self.vocab.items():
            symbols = word.split()

            for i in range(1, len(symbols)):
                pairs[symbols[i-1], symbols[i]] += frequency
            
        return pairs


    def _merge_vocab(self, char_pair_to_merge, vocab):
        
        modified_vocab = {}

        bigram = re.escape(' '.join(char_pair_to_merge))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in vocab:
            # replace most frequent pair in all vocabulary
            w_out = p.sub(''.join(char_pair_to_merge), word)
            # print(''.join(char_pair_to_merge), w_out)
            modified_vocab[w_out] = vocab[word]
        
        return modified_vocab
    

    def _save_scatter_plot(self, descriptive_name, x, y):

        plt.scatter(x, y)
        plt.title('BPE Algorithm')
        plt.xlabel('Size of Type Vocabulary')
        plt.ylabel('Length of training corpus (tokens)')
        plt.savefig(f'{descriptive_name}_bpe_scatterplot.png')


    def get_data_lengths(self):
        type_vocab = set()
        corpus_length = 0

        for k, v in self.vocab.items():
            merged = k.split(" ")
            type_vocab.update(merged)
            corpus_length += (len(merged) * v)
        
        self.types_vocab = self.types_vocab.union(type_vocab)
        return len(self.types_vocab), corpus_length


    def _save_information(self, descriptive_name, x, y, num_iters, merge_pairs_list):
        xx, yy = self.get_data_lengths()

        with open(f"{descriptive_name}.out", 'w') as f:
            f.write(f'Number of Iterations Completed: {num_iters}\n\n')
            f.write(f'Length of Types Vocab: {xx}\n')
            f.write(f'Frequency Sum of Types Vocab: {sum(self.char_pairs.values())}\n\n')
            f.write(f'Length of Training Vocab: {len(self.vocab)}\n')
            f.write(f'Frequency of Training Vocab: {sum(self.vocab.values())}\n')
            f.write(f'Length of Training Data Under Types: {yy}\n\n')

        with open(f"{descriptive_name}_merge_rules.out", 'w') as f:
            f.writelines([f"{item[0]}||{item[1]}\n" for item in merge_pairs_list])
        
        def remap_keys(mapping):
            return [{'key':k, 'value': v} for k, v in mapping.items()]
        
        with open(f'char_pairs_{descriptive_name}.json', 'w') as f:
            json.dump(remap_keys(self.char_pairs), f, indent=4, sort_keys=True)

        with open(f'vocab_final_{descriptive_name}.json', 'w') as f:
            json.dump(self.vocab, f, indent=4, sort_keys=True)
        
        self._save_scatter_plot(descriptive_name, x, y)


    def encode_word(self, ordered_merge_pairs_path, word):
        ordered_merge_pairs = []
        with open(ordered_merge_pairs_path, 'r') as f:
            ordered_merge_pairs = f.readlines()
        
        ordered_merge_pairs = self._clean_sentences(ordered_merge_pairs)

        word_l = self._clean_sentences([word])
        vocab, word_l = self._add_word_space_symbols(word_l)

        for pair in ordered_merge_pairs:
            pair = pair.split("||")
            vocab = self._merge_vocab(pair, vocab)
        
        return list(vocab.keys())[0].split(' ')


    def fit(self):

        with open('vocab_initial.json', 'w') as f:
            json.dump(self.vocab, f, indent=4, sort_keys=True)

        x, y = [], []
        merge_pairs_list = []

        i = 0
        is_frequency_one = False
        while True:
            i += 1

            # Step 2: Count the character pairs
            self.char_pairs = self._get_pair_counts()
            
            # Used to produce a scatter plot of the algorithm
            xx, yy = self.get_data_lengths()
            x.append(xx)
            y.append(yy)

            # Indicates we are done with fitting.
            # No more merge rules can be applied here
            if not self.char_pairs: break
            
            top_char_pair = self._get_most_frequent_bigram(self.char_pairs)
            top_pair_frequency = self.char_pairs[top_char_pair]
            merge_pairs_list.append(top_char_pair)

            # Track output
            if i % 1000: print(f"Current top pair frequency: {top_pair_frequency}")

            if top_pair_frequency == 1 and not is_frequency_one:
                self._save_information('frequency_one', x, y, i, merge_pairs_list)
                is_frequency_one = True

            self.vocab = self._merge_vocab(top_char_pair, self.vocab)
            
            # print(f"Bigram to Merge: {top_char_pair}")
            # pp(self.char_pairs)
            # pp(self.vocab)
            # print(self.get_data_lengths())
            # print(self.types_vocab)
            # print()

        self._save_information('frequency_zero', x, y, i, merge_pairs_list)


    def _get_most_frequent_bigram(self, char_pairs):
        return max(char_pairs, key=char_pairs.get)


    def _flatten(self, l: list):
        flat_list = [item for sublist in l for item in sublist]
        return flat_list


if __name__ == "__main__":
    print("A5 - NLP - Apoorv Sharma")

    bpe = BytePairEncoding("A5-data.txt")
    bpe.fit()
    
    with open(f"frequency_one.out", 'a') as f:
        f.writelines([f"{item} " for item in bpe.encode_word('frequency_one_merge_rules.out', 'Acnestis')])

