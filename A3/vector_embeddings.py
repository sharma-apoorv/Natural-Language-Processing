''' Assignment 3 - NLP

Author: Apoorv Sharma
Description:
'''

# Import common required python files
import os
import random
import re
from collections import Counter

# Import special libraries
import numpy as np

import nltk
nltk.download('punkt')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters
MAX_SEQ_LEN = -1  # -1 for no truncation
UNK_THRESHOLD = 5
BATCH_SIZE = 128
N_EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_RNN_LAYERS = 2

# Data cleaning parms
L_RAW, L_LABEL, L_TOKENS = "raw", "label", "text"
PAD = "@@PAD@@"
UNK = "@@UNK@@"

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

        self.train_files_list, self.dev_files_list, self.test_files_list = [], [], []

        # Generate a list of test files
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_TRAIN_FILES)
        self.train_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        # Generate a list of dev files
        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        random_files_index = random.sample(range(len(movie_review_file_list)), NUM_DEV_FILES)
        self.dev_files_list = [movie_review_file_list[i] for i in random_files_index] # train files list

        for index in sorted(random_files_index, reverse=True): del movie_review_file_list[index]
        self.test_files_list = movie_review_file_list[:]

    def _apply_vocab(self, data):
        """
        Applies the vocabulary to the data and maps the tokenized sentences to vocab indices as the
        model input.
        """
        for review in data:
            review[L_TOKENS] = [self.token_to_index.get(token, self.token_to_index[UNK]) for token in review[L_TOKENS]]

    def _create_vocab(self, data, unk_threshold=UNK_THRESHOLD):
        """
        Creates a vocabulary with tokens that have frequency above unk_threshold and assigns each token
        a unique index, including the special tokens.
        """
        counter = Counter(token for review in data for token in review[L_TOKENS])
        vocab = {token for token in counter if counter[token] > unk_threshold}
        # print(f"Vocab size: {len(vocab) + 2}")  # add the special tokens
        # print(f"Most common tokens: {counter.most_common(10)}")
        token_to_idx = {PAD: 0, UNK: 1}
        for token in vocab:
            token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def _create_dataset(self, files_list):

        data = []
        for file in files_list:
            
            # Read the file to analyze
            with open(file) as f:
                label = -1
                if file.split("/")[1] == "pos": label = self.POS
                elif file.split("/")[1] == "neg": label = self.NEG

                data.append({
                    L_RAW: f.read(),
                    L_LABEL: label
                })
        
        return data
            
    def _tokenize(self, data, max_seq_len=MAX_SEQ_LEN):
        """
        Here we use nltk to tokenize data. There are many other possibilities. We also truncate the
        sequences so that the training time and memory is more manageable. You can think of truncation
        as making a decision only looking at the first X words.

        Reference: CSE-447 Section AB
        """
        for review in data:
            review[L_TOKENS] = []
            for sent in nltk.sent_tokenize(review[L_RAW]):
                review[L_TOKENS].extend(nltk.word_tokenize(sent))
            if max_seq_len >= 0:
                review[L_TOKENS] = review[L_TOKENS][:max_seq_len]

    def pre_process_data(self):
        # Create a dictionary with labels and raw text for each dataset
        self.train_dataset, self.dev_dataset, self.test_dataset = [], [], []
        self.POS, self.NEG = 1, 0

        # Create a list of information for file to parse
        self.train_dataset = self._create_dataset(self.train_files_list)
        self.dev_dataset = self._create_dataset(self.dev_files_list)
        self.test_dataset = self._create_dataset(self.test_files_list)

        # Tokenize the all the 'raw' sentences
        for data in (self.train_dataset, self.dev_dataset, self.test_dataset):
            self._tokenize(data)

        # Map the vocab to an index
        self.token_to_index = self._create_vocab(self.train_dataset)

        for data in (self.train_dataset, self.dev_dataset, self.test_dataset):
            self._apply_vocab(data)
    
    def get_token_to_index_mapping(self):
        return self.token_to_index
    
    def get_train_data(self):
        return self.train_dataset
    
    def get_dev_data(self):
        return self.dev_dataset
    
    def get_test_data(self):
        return self.test_dataset

class SentimentDataset(Dataset):
    def __init__(self, data, pad_idx):
        data = sorted(data, key=lambda review: len(review[L_TOKENS]))
        self.texts = [review[L_TOKENS] for review in data]
        self.labels = [review[L_LABEL] for review in data]
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index]]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        def tensorize(elements, dtype):
            return [torch.tensor(element, dtype=dtype) for element in elements]

        def pad(tensors):
            """Assumes 1-d tensors."""
            max_len = max(len(tensor) for tensor in tensors)
            padded_tensors = [
                F.pad(tensor, (0, max_len - len(tensor)), value=self.pad_idx) for tensor in tensors
            ]
            return padded_tensors

        texts, labels = zip(*batch)
        return [
            torch.stack(pad(tensorize(texts, torch.long)), dim=0),
            torch.stack(tensorize(labels, torch.long), dim=0),
        ]

class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, pad_idx):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers=n_rnn_layers, batch_first=True, bidirectional=True
        )
        # We take the final hidden state at all GRU layers as the sequence representation.
        # 2 because bidirectional.
        layered_hidden_dim = hidden_dim * n_rnn_layers * 2
        self.output = nn.Linear(layered_hidden_dim, n_labels)

    def forward(self, text):
        # text shape: (batch_size, max_seq_len) where max_seq_len is the max length *in this batch*
        # lens shape: (batch_size,)
        non_padded_positions = text != self.pad_idx
        lens = non_padded_positions.sum(dim=1)

        # embedded shape: (batch_size, max_seq_len, embedding_dim)
        embedded = self.embedding(text)
        # You can pass the embeddings directly to the RNN, but as the input potentially has
        # different lengths, how do you know when to stop unrolling the recurrence for each example?
        # pytorch provides a util function pack_padded_sequence that converts padded sequences with
        # potentially different lengths into a special PackedSequence object that keeps track of
        # these things. When passing a PackedSequence object into the RNN, the output will be a
        # PackedSequence too (but not the hidden state as that always has a length of 1). Since we
        # do not use the per-token output, we do not unpack it. But if you need it, e.g. for
        # token-level classification such as POS tagging, you can use pad_packed_sequence to convert
        # it back to a regular tensor.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # nn.GRU produces two outputs: one is the per-token output and the other is per-sequence.
        # The pers-sequence output is simiar to the last per-token output, except that it is taken
        # at all layers.
        # output (after unpacking) shape: (batch_size, max_seq_len, hidden_dim)
        # hidden shape: (n_layers * n_directions, batch_size, hidden_dim)
        packed_output, hidden = self.rnn(packed_embedded)
        # shape: (batch_size, n_layers * n_directions * hidden_dim)
        hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)
        # Here we directly output the raw scores without softmax normalization which would produce
        # a valid probability distribution. This is because:
        # (1) during training, pytorch provides a loss function "F.cross_entropy" that combines
        # "log_softmax + F.nll_loss" in one step. See the `train` function below.
        # (2) during evaluation, we usually only care about the class with the highest score, but
        # not the actual probablity distribution.
        # shape: (batch_size, n_labels)
        return self.output(hidden)

def word_embedding_questions(glove, output_file):

    _num_closest_words = 1

    output_file.write(f"*********Question 3.1*********\n")

    find_closest_words_list = ["dog", "whale", "before", "however", "fabricate"]
    for word in find_closest_words_list:
        output_file.write(f"{word}: {glove.get_x_closest_words(word, num_closest_words=_num_closest_words)}\n")
    
    output_file.write(f"\n*********Question 3.2*********\n")

    word_analogy_list = [("dog", "puppy", "cat"), ("speak", "speaker", "sing"), ("france", "french", "england"), ("france", "wine", "england")]
    for w1, w2, w3 in word_analogy_list:
        output_file.write(f"{w1} : {w2} :: {w3} : {glove.get_word_analogy_closest_word(w1, w2, w3, num_closest_words=_num_closest_words)}\n")

def train(model, dataloader, optimizer, device):
    for texts, labels in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device):
    count = correct = 0.0
    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            # shape: (batch_size, n_labels)
            output = model(texts)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1)
            count += len(predicted)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / count}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sentiment_classification(output_file):
    cmr = CornellMovieReviewFiles()
    cmr.pre_process_data()
    
    token_to_index_mapping = cmr.get_token_to_index_mapping()
    train_data, dev_data, test_data = cmr.get_train_data(), cmr.get_dev_data(), cmr.get_test_data()

    pad_idx = token_to_index_mapping[PAD]
    label_to_idx = {"neg": 0, "pos": 1}
    
    # Create instances of dataset classes
    train_dataset = SentimentDataset(train_data, pad_idx)
    dev_dataset = SentimentDataset(dev_data, pad_idx)
    test_dataset = SentimentDataset(test_data, pad_idx)

    # Not sure what this is exactly ... ?
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, collate_fn=dev_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=test_dataset.collate_fn
    )

    model = SequenceClassifier(
        len(token_to_index_mapping), EMBEDDING_DIM, HIDDEN_DIM, len(label_to_idx), N_RNN_LAYERS, pad_idx
    )
    print(f"Model has {count_parameters(model)} parameters.")

    # Adam is just a fancier version of SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Random baseline")
    evaluate(model, dev_dataloader, device)
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}")  # 0-based -> 1-based
        train(model, train_dataloader, optimizer, device)
        evaluate(model, dev_dataloader, device)
    print(f"Test set performance")
    evaluate(model, test_dataloader, device)

if __name__ == '__main__':
    # glove = GloVeWordEmbeddings('glove.42B.300d.txt', 300)

    output_file_name = "output.txt"
    output_file = open(output_file_name, "w")

    # Question 3.1 and 3.2
    # word_embedding_questions(glove, output_file)

    # Question 3.3
    sentiment_classification(output_file_name)


    print(f"Done! Check {output_file_name} for details.")