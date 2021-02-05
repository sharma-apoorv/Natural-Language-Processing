import numpy as np

def relu(x):
    return np.maximum(x, 0)

def score_q2(theta, word_emb_list):
    score = 0
    word_emb_sum = relu(sum(word_emb_list))
    score = np.dot(theta, word_emb_sum)
    return score

_good = np.array([3, 1, 2, -4])
_bad = np.array([0.5, 0.5, 0.5, -1])
_not = np.array([-3, -1, -2, 4])

theta = np.array([3, 4, 4, 20])

print(f"good: {score_q2(theta, [_good])}")
print(f"not good: {score_q2(theta, [_not, _good])}")

print(f"bad: {score_q2(theta, [_bad])}")
print(f"not bad: {score_q2(theta, [_not, _bad])}")


# Archived code
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

    def pre_process_data(self, _token_to_idx=None):
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
        self.token_to_index = _token_to_idx
        if not _token_to_idx:
            # Only build vocab from current vocab, if not vector embeddings specified
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