# Vector Embeddings 

## Introduction

Word embedding is any of a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with many dimensions per word to a continuous vector space with a much lower dimension.

This assignment illustrates the use of GloVe word embeddings to do the following:

1. Find synonyms
2. Find word analogies
3. Predict sentiment of movie reviews using word embeddings and a neural net model

## Description

The code contains 4 main class definitions:
1. `GloVeWordEmbeddings`
    * This class contains the code to parse the word embeddings file 
2. `IMDBMovieReviews`
    * This class contains the code to download movie review data and parse the reviews so that this can be input into the neural net model.
3. `SentimentDataset`
    * This class is an inherited from `Dataset` class in PyTorch. It is used to fine tune the model and pad the data
4. `SequenceClassifier`
    * This class contains the neural net model

### Find Synonyms

The first part of the assignment was to use the cosine similarity (1 - cosine distance) to find the closest 'x' matches to a specified word. The function `get_x_closest_words()` shows how this can be using the cosine distance. The following is the process to find the closest words:

1. Ensure the word exists in the word embeddings
2. Find the cosine similarity to all other words
3. Sort the keys (words) based on the similarity (1 = complete match / -1 = No match)
4. Return the top 'x' words

### Find Word Analogy

The second part of the assignment is to find word analogies. In the broadest sense, a word analogy is a statement of the form “a is to b as x is to y”, which asserts that a and x can be transformed in the same way to get b and y, and vice-versa. The form of the input is as follows:

dog :: puppy : cat :: ?

The above statement means 'dog is to puppy as cat is to ... ?'

The following is the code implementation:
1. Ensure all 3 words are in the word embeddings
2. Compute `w2 - w1 + w3` where (w1 = dog, w2 = puppy, w3 = cat)
3. Return the top 'x' choices for each analogy

### Neural Net Model

A GRU-based RNN was used as the model. The base code used was provided as part of the CSE 447 section course material. Further modifications were made to then adhere to the specifications of the assignments. The following table denotes the hyper parameters being used:

1. MAX_SEQ_LEN = 20000
2. UNK_THRESHOLD	5
3. BATCH_SIZE	128
4. N_EPOCHS	7
5. LEARNING_RATE	0.001
6. HIDDEN_DIM	256
7. N_RNN_LAYERS	2

The follow code get executed when the model needs to be trained:
1. Download IMDB Movie Review Data
2. Tokenize the data
3. Create a word to index mapping for the words (UNK and PAD are used here to handle OOV words)
4. Convert the word to index values (based on mapping)
5. Initialize the dataset class
6. initialize the dataloader class
7. Train the model
    * Here words that are in the word embeddings have their gradients zero'd out (depending on configuration)
8. Test the model on the test dataset

## Usage / Running the code

The file assumes the following directory structure and files to be present:

```
├── vector_embeddings.py            # The source code file
├── glove.6B.50d.txt                # Some vector embedding file
├── glove.6B.100d.txt
├── glove.42B.300d.txt
├── glove.6B.200d.txt
└── glove.6B.300d.txt
```

The model can be trained and tested using the following command:
```
python3 vector_embeddings.py -g [VECTOR_EMBEDDING_FILE]
```

For example:
```
python3 vector_embeddings.py -g glove.6B.50d.txt
```

**Note, the `VECTOR_EMBEDDING_FILE` must have the following form:**
`something_something.50d.txt`

The following code snippet is used to find the number of dimensions from the file 
```
int((args.glove_path.split(".")[-2]).split("d")[0])
```
Thus, the file needs to end in **\*.50d.txt** 

## References

1. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.

2. CSE 447 - Course Material

3. Large Movie Review Dataset (Stanford)