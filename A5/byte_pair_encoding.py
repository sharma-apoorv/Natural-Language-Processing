import os
import string
from collections import Counter, defaultdict
import re

#debug imports
from pprint import pprint as pp
import json

SPACE_SYM = '<s>'

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

        # def remap_keys(mapping):
        #     return [{'key':k, 'value': v} for k, v in mapping.items()]
        # with open('char_pairs.json', 'w') as f:
        #     json.dump(remap_keys(self.char_pairs), f)
    

    def _clean_sentences(self, sentences_list):
        sentences_list = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences_list]
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


    def _merge_vocab(self, char_pair_to_merge):
        
        modified_vocab = {}

        bigram = re.escape(' '.join(char_pair_to_merge))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in self.vocab:
            # replace most frequent pair in all vocabulary
            w_out = p.sub(''.join(char_pair_to_merge), word)
            modified_vocab[w_out] = self.vocab[word]
        
        return modified_vocab
    

    
    def fit(self):

        with open('vocab_initial.json', 'w') as f:
            json.dump(self.vocab, f, indent=4, sort_keys=True)

        while True:
            # Step 2: Count the character pairs
            self.char_pairs = self._get_pair_counts()

            # Indicates we are done with fitting.
            # No more merge rules can be applied here
            if not self.char_pairs: break
            
            top_char_pair = self._get_most_frequent_bigram(self.char_pairs)
            self.vocab = self._merge_vocab(top_char_pair)

        with open('vocab_final.json', 'w') as f:
            json.dump(self.vocab, f, indent=4, sort_keys=True)


    def _get_most_frequent_bigram(self, char_pairs):
        return max(char_pairs, key=char_pairs.get)


    def _flatten(self, l: list):
        flat_list = [item for sublist in l for item in sublist]
        return flat_list


if __name__ == "__main__":
    print("A5 - NLP - Apoorv Sharma")

    bpe = BytePairEncoding("A5-data.txt")
    bpe.fit()

