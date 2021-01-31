''' Assignment 3 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import re
import math

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

if __name__ == '__main__':
    print("CSE 447 - NLP")
    glove = GloVeWordEmbeddings('glove.42B.300d.txt', 300)

    output_file_name = "output.txt"
    output_file = open(output_file_name, "w")

    _num_closest_words = 1

    output_file.write(f"*********Question 3.1*********\n")

    find_closest_words_list = ["dog", "whale", "before", "however", "fabricate"]
    for word in find_closest_words_list:
        output_file.write(f"{word}: {glove.get_x_closest_words(word, num_closest_words=_num_closest_words)}\n")
    
    output_file.write(f"\n*********Question 3.2*********\n")

    word_analogy_list = [("dog", "puppy", "cat"), ("speak", "speaker", "sing"), ("France", "French", "England"), ("France", "wine", "England")]
    for w1, w2, w3 in word_analogy_list:
        output_file.write(f"{w1} : {w2} :: {w3} : {glove.get_word_analogy_closest_word(w1, w2, w3, num_closest_words=_num_closest_words)}\n")

    output_file.close()