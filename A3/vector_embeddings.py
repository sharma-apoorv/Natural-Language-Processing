''' Assignment 3 - NLP

Author: Apoorv Sharma
Description:
'''

import argparse
import os
import random
import re
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nltk.download('punkt')

# Hyperparameters
MAX_SEQ_LEN = 20000  # -1 for no truncation
UNK_THRESHOLD = 5
BATCH_SIZE = 128
N_EPOCHS = 7
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256
N_RNN_LAYERS = 2

# Data cleaning variables
L_RAW, L_LABEL, L_TOKENS = "raw", "label", "text"
PAD = "@@PAD@@"
UNK = "@@UNK@@"

def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class GloVeWordEmbeddings():
    def __init__(self, glove_file_path, num_dims):
        # The number of dimensions contained in the glove vector embeddings
        self.num_dims = num_dims

        # Ensure we have a valid file
        if not os.path.exists(glove_file_path):
            print("Error! Not a valid glove path")
            return
        
        self.token_to_embedding = {
            PAD: np.random.normal(size=(num_dims, )),
            UNK: np.random.normal(size=(num_dims, ))
        }

        ''' Parse file and create the following map:
            word -> vector embedding
        '''
        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()

                # create a dict of word -> positions
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                
                self.token_to_embedding[word] = vector
    
    def get_token_to_embedding(self):
        return self.token_to_embedding
    
    def get_num_dims(self):
        return self.num_dims

    def _get_cosine_similarity(self, vecA: np.array, vecB: np.array):
        return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

    def _get_closest_words(self, embedding):
        return sorted(self.token_to_embedding.keys(), key=lambda w: self._get_cosine_similarity(self.token_to_embedding[w], embedding), reverse=True)
    
    def _get_embedding_for_word(self, word: str) -> np.array:
        if word in self.token_to_embedding.keys():
            return self.token_to_embedding[word]
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
        closest_words = self._get_closest_words(embedding)
        for w in [word, PAD, UNK]: closest_words.remove(w)

        return closest_words[:num_closest_words]
    
    def get_word_analogy_closest_words(self, w1, w2, w3, num_closest_words=1):
        e1 = self._get_embedding_for_word(w1)
        e2 = self._get_embedding_for_word(w2)
        e3 = self._get_embedding_for_word(w3)

        if e1.size == 0 or e2.size == 0 or e3.size == 0:
            print(f"{w1}:{e1.size}  {w2}:{e2.size}  {w3}:{e3.size}")
            return []

        embedding = e2 - e1 + e3
        closest_words = self._get_closest_words(embedding)
        for w in [w1, w2, w3, PAD, UNK]: closest_words.remove(w) # We dont want the have the same words in the output
        return closest_words[:num_closest_words]

class IMDBMovieReviews():
    '''
    This code the creation and parsing of this dataset has been taken from
    CSE 447 (UW) NLP course material. 
    '''

    def __init__(self):
        return

    def download_data(self):
        def extract_data(dir, split):
            data = []
            for label in ("pos", "neg"):
                label_dir = os.path.join(dir, "aclImdb", split, label)
                files = sorted(os.listdir(label_dir))
                for file in files:
                    filepath = os.path.join(label_dir, file)
                    with open(filepath, encoding="UTF-8") as f:
                        data.append({L_RAW: f.read(), L_LABEL: label})
            return data

        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        stream = urllib.request.urlopen(url)
        tar = tarfile.open(fileobj=stream, mode="r|gz")
        with tempfile.TemporaryDirectory() as td:
            tar.extractall(path=td)
            train_data = extract_data(td, "train")
            test_data = extract_data(td, "test")
            return train_data, test_data

    def split_data(self, train_data, num_split=2000):
        """Splits the training data into training and development sets."""
        random.shuffle(train_data)
        return train_data[:-num_split], train_data[-num_split:]

    def tokenize(self, data, max_seq_len=MAX_SEQ_LEN):
        """
        Here we use nltk to tokenize data. There are many other possibilities. We also truncate the
        sequences so that the training time and memory is more manageable. You can think of truncation
        as making a decision only looking at the first X words.
        """
        for review in data:
            review[L_TOKENS] = []
            for sent in nltk.sent_tokenize(review[L_RAW]):
                review[L_TOKENS].extend(nltk.word_tokenize(sent))
            if max_seq_len >= 0:
                review[L_TOKENS] = review[L_TOKENS][:max_seq_len]

    def create_vocab(self, data, unk_threshold=UNK_THRESHOLD):
        """
        Creates a vocabulary with tokens that have frequency above unk_threshold and assigns each token
        a unique index, including the special tokens.
        """
        counter = Counter(token for review in data for token in review[L_TOKENS])
        self.vocab = {token for token in counter if counter[token] > unk_threshold}
        token_to_idx = {PAD: 0, UNK: 1}
        for token in self.vocab:
            token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def get_embeds(self, token_to_index_mapping, token_to_glove, dim):
        weights_matrix = np.zeros((len(token_to_index_mapping), dim))
        indices_found = []

        for word, i in token_to_index_mapping.items():
            if word in token_to_glove.keys():
                indices_found.append(i) # This gradient of these indices will get zero'd out later depending on config
            weights_matrix[i] = token_to_glove.get(word, np.random.normal(size=(dim, )))
        return indices_found, weights_matrix

    def apply_vocab(self, data, token_to_idx):
        """
        Applies the vocabulary to the data and maps the tokenized sentences to vocab indices as the
        model input.
        """
        for review in data:
            review[L_TOKENS] = [token_to_idx.get(token, token_to_idx[UNK]) for token in review[L_TOKENS]]


    def apply_label_map(self, data, label_to_idx):
        """Converts string labels to indices."""
        for review in data:
            review[L_LABEL] = label_to_idx[review[L_LABEL]]

class SentimentDataset(Dataset):
    '''
    The initial code was taken from CSE 447 (UW) NLP course material.
    It was then modified to serve the required specifications.
    '''

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
    '''
    The initial code was taken from CSE 447 (UW) NLP course material.
    It was then modified to serve the required specifications.
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, pad_idx, embedding_matrix, freeze=True):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # self.embedding.weight.requires_grad = not freeze # to train or freeze the embeddings
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
        # different lengths, how do you know when to stop unrolling the recurrence for each review?
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
        # The per-sequence output is similar to the last per-token output, except that it is taken
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
        # not the actual probability distribution.
        # shape: (batch_size, n_labels)
        return self.output(hidden)

def word_embedding_questions(glove, output_file):

    _num_closest_words = 3

    output_file.write(f"*********Question 3.1*********\n")

    find_closest_words_list = ["dog", "whale", "before", "however", "fabricate"]
    for word in find_closest_words_list:
        output_file.write(f"{word}: {glove.get_x_closest_words(word, num_closest_words=_num_closest_words)}\n")
    
    output_file.write(f"\n*********Question 3.2*********\n")

    word_analogy_list = [("dog", "puppy", "cat"), ("speak", "speaker", "sing"), ("france", "french", "england"), ("france", "wine", "england")]
    for w1, w2, w3 in word_analogy_list:
        output_file.write(f"{w1} : {w2} :: {w3} : {glove.get_word_analogy_closest_words(w1, w2, w3, num_closest_words=_num_closest_words)}\n")

def train(model, dataloader, optimizer, device, indices_found, word_embeddings_freeze=True):
    for texts, labels in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()

        # Zero out the gradients of the word embeddings that we found
        if word_embeddings_freeze:
            model.embedding.weight.grad[indices_found] = 0

        optimizer.step()

def evaluate(model, dataloader, device):
    count = correct = 0.0

    y_pred, y_labels = [], []
    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            # shape: (batch_size, n_labels)
            output = model(texts)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1)
            count += len(predicted)
            correct += (predicted == labels).sum().item()

            y_pred.extend(predicted.tolist())
            y_labels.extend(labels.tolist())
    
    f1_score_macro = f1_score(y_labels, y_pred, average='macro')

    accuracy = (correct / count) * 100
    print(f"Accuracy: {accuracy} %")

    return accuracy, f1_score_macro

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sentiment_classification(glove, output_file):
    imdb_reviews = IMDBMovieReviews()
    
    print("Downloading data")

    # Download the data and convert this into a dictionary format for each review in the dataset
    train_data, test_data = imdb_reviews.download_data()
    train_data, dev_data = imdb_reviews.split_data(train_data)

    print("Processing data")
    
    # Begin by coverting the text into single words
    for data in (train_data, dev_data, test_data):
        imdb_reviews.tokenize(data) 

    # Get the metadata, used later for model creation and initialization
    token_to_index_mapping = imdb_reviews.create_vocab(train_data)
    token_to_glove_mapping = glove.get_token_to_embedding()
    indices_found, embedding_matrix = imdb_reviews.get_embeds(token_to_index_mapping, token_to_glove_mapping, glove.get_num_dims())

    # Convert the vocab for each movie review into the mapping obtained earlier
    label_to_idx = {"neg": 0, "pos": 1}
    for data in (train_data, dev_data, test_data):
        imdb_reviews.apply_vocab(data, token_to_index_mapping)
        imdb_reviews.apply_label_map(data, label_to_idx)

    # Create instances of dataset classes
    pad_idx = token_to_index_mapping[PAD]
    train_dataset = SentimentDataset(train_data, pad_idx)
    dev_dataset = SentimentDataset(dev_data, pad_idx)
    test_dataset = SentimentDataset(test_data, pad_idx)

    # Data load to process the data in batches
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, collate_fn=dev_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=test_dataset.collate_fn
    )

    output_file.write(f"\n*********Question 3.4*********\n")
    print("Starting Training - Frozen Vector Embeddings Model")
    # Model creation and initialization - This model will freeze the word embeddings!
    model_freeze = SequenceClassifier(
        len(token_to_index_mapping), # vocab size (based on training set)
        glove.get_num_dims(), # the number of dimensions for the vector embeddings
        HIDDEN_DIM, 
        len(label_to_idx), # number of outputs 
        N_RNN_LAYERS, 
        pad_idx, # Index to the pad the data with to make all inputs equal sizes
        embedding_matrix, # The embedding matrix. Note: The index of this and token_to_index_mapping must align!
        freeze = True # Wean
    )
    print(f"Freeze Model has {count_parameters(model_freeze)} parameters.")

    # Adam is just a fancier version of SGD.
    optimizer = torch.optim.Adam(model_freeze.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_freeze.to(device)

    start_time = time.time()
    accuracy_list = []
    print(f"Random baseline")
    evaluate(model_freeze, dev_dataloader, device)
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}")  # 0-based -> 1-based
        train(model_freeze, train_dataloader, optimizer, device, indices_found, word_embeddings_freeze=True)
        accuracy_list.append(evaluate(model_freeze, dev_dataloader, device)[0])
    elapsed_time_fl = (time.time() - start_time)
    output_file.write(f"Finished training in {elapsed_time_fl:.2f} seconds\n")
    
    print(f"Test set performance")
    output_file.write(f"Dev Accuracy (Freeze Model)\n")
    for accuracy in accuracy_list:
        output_file.write(f"{accuracy:.2f} ")
    output_file.write(f"\n")
    accu, f1_score = evaluate(model_freeze, test_dataloader, device)
    output_file.write(f"Test Accuracy (Freeze Model): {accu:.2f}\n")
    output_file.write(f"Test Macro F-1 Score (Freeze Model): {f1_score:.2f}\n")

    output_file.write(f"\n*********Question 3.5*********\n")
    print("Starting Training - Fine Tune Embeddings Model")
    # Model creation and initialization - This model will NOT freeze the word embeddings!
    model_finetune = SequenceClassifier(
        len(token_to_index_mapping), # vocab size (based on training set)
        glove.get_num_dims(), # the number of dimensions for the vector embeddings
        HIDDEN_DIM, 
        len(label_to_idx), # number of outputs 
        N_RNN_LAYERS, 
        pad_idx, # Index to the pad the data with to make all inputs equal sizes
        embedding_matrix, # The embedding matrix. Note: The index of this and token_to_index_mapping must align!
        freeze = False # Wean
    )
    print(f"Fine Tune Model has {count_parameters(model_finetune)} parameters.")

    # Adam is just a fancier version of SGD.
    optimizer = torch.optim.Adam(model_finetune.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_finetune.to(device)

    start_time = time.time()
    accuracy_list = []
    print(f"Random baseline")
    evaluate(model_finetune, dev_dataloader, device)
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}")  # 0-based -> 1-based
        train(model_finetune, train_dataloader, optimizer, device, indices_found, word_embeddings_freeze=False)
        accuracy_list.append(evaluate(model_finetune, dev_dataloader, device)[0])
    
    elapsed_time_fl = (time.time() - start_time)
    output_file.write(f"Finished training in {elapsed_time_fl:.2f} seconds\n")
    
    print(f"Test set performance")
    output_file.write(f"Dev Accuracy (Fine Tune Model)")
    for accuracy in accuracy_list:
        output_file.write(f"{accuracy:.2f} ")
    output_file.write(f"\n")
    accu, f1_score = evaluate(model_finetune, test_dataloader, device)
    output_file.write(f"Test Accuracy (ine Tune Model): {accu:.2f}\n")
    output_file.write(f"Test Macro F-1 Score (ine Tune Model): {f1_score:.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vector Embeddings')
    parser.add_argument("-g", "--word-emb", dest="glove_path", type=str, required=True)
    args = parser.parse_args()

    seed_everything()

    glove = GloVeWordEmbeddings(args.glove_path, int((args.glove_path.split(".")[-2]).split("d")[0]))

    output_file_name = "output.txt"
    output_file = open(output_file_name, "w")

    # Question 3.1 and 3.2
    # word_embedding_questions(glove, output_file)

    # Question 3.3, 3.4, 3.5
    sentiment_classification(glove, output_file)

    print(f"Done! Check {output_file_name} for details.")

    output_file.close()
