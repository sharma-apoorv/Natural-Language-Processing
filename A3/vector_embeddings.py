''' Assignment 3 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import random
import re

# Import speical libraries
import numpy as np

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

        # Generate a list of test files
        self.train_files_list, self.dev_files_list, self.test_files_list = [], [], []
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_TRAIN_FILES)
        self.train_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        # Generate a list of dev files
        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_DEV_FILES)
        self.dev_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        self.test_files_list = movie_review_file_list[:]
    
    def _tokenize_clean(self, text: str) -> list:
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
    
    def _get_file_tokens(self, file_list, clean_tokens=False):
        tokens = []
        for file in file_list:
        
            with open(file) as f:
                sentences = f.readlines()

            for sentence in sentences:
                if clean_tokens: tokens += self._tokenize_clean(sentence)
                else: 
                    for token in sentence.split(): tokens.append(token)
        
        return tokens

    def get_train_file_tokens(self, clean_tokens=False):
        self._get_file_tokens(self.train_files_list, clean_tokens)
    
    def get_dev_file_tokens(self, clean_tokens=False):
        self._get_file_tokens(self.dev_files_list, clean_tokens)
    
    def get_test_file_tokens(self, clean_tokens=False):
        self._get_file_tokens(self.test_files_list, clean_tokens)


def word_embedding_questions(glove, output_file):

    _num_closest_words = 1

    output_file.write(f"*********Question 3.1*********\n")

    find_closest_words_list = ["dog", "whale", "before", "however", "fabricate"]
    for word in find_closest_words_list:
        output_file.write(f"{word}: {glove.get_x_closest_words(word, num_closest_words=_num_closest_words)}\n")
    
    output_file.write(f"\n*********Question 3.2*********\n")

    word_analogy_list = [("dog", "puppy", "cat"), ("speak", "speaker", "sing"), ("France", "French", "England"), ("France", "wine", "England")]
    for w1, w2, w3 in word_analogy_list:
        output_file.write(f"{w1} : {w2} :: {w3} : {glove.get_word_analogy_closest_word(w1, w2, w3, num_closest_words=_num_closest_words)}\n")

if __name__ == '__main__':
    print("CSE 447 - NLP")
    # glove = GloVeWordEmbeddings('glove.42B.300d.txt', 300)

    output_file_name = "output.txt"
    output_file = open(output_file_name, "w")

    # Question 3.1 and 3.2
    # word_embedding_questions(glove, output_file)

    # Question 3.3
    cmr = CornellMovieReviewFiles()
    cmr.get_test_file_tokens(clean_tokens=False)
    cmr.get_test_file_tokens(clean_tokens=True)
    
    output_file.close()