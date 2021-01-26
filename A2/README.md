# Language Models

## Introduction

This project implements n-gram models (unigram, bigram and trigram) as well as interpolation smoothing for the trigram model.

## Description

The source code is contained in the `n-gram-models.py` file.

The source code implements 4 classes:
1. `FileParser()`
2. `UnigramLanguageModel()`
3. `BigramLanguageModel(UnigramLanguageModel)`
4. `TrigramLanguageModel(BigramLanguageModel)`

### `FileParser()`

This class is responsible for parsing the test/validation/test datasets. The functions implemented in the class do basic tokenization of the text (converting to words) and return this as a list of words for the respective file. 

### `UnigramLanguageModel()`

This class implements a unigram model. Upon initilization, this class builds a frequency dictionary for the tokens and also does some *unkinization* for the words.

#### Probability Calculation

The probability for a token in the unigram model is calculated as follows: 

**frequency_of_word / corpus_length**

If the word does not exist, the frequency of the number of `UNK` tokens is used.

#### Perplexity Calculation

The perplexity for a token in the unigram model is calculated as follows:

**-1 x sum(log(probability_of_all_words_in_testset)) / number_of_words_in_testset**

### `BigramLanguageModel()`

This class implements a bigram model. Upon initilization, this class first initilizes the `UnigramLanguageModel()` and then builds a frequency dictionary for all the bigrams in the train dataset.

#### Probability Calculation

The probability for a token in the bigram model is calculated as follows: 

**count(hv) / count(h)** where, h is the history (the previous word) and v is the current word.

#### Perplexity Calculation

The perplexity for a token in the bigram model is calculated as follows:

**-1 x sum(log(probability_of_all_bigrams_in_testset)) / number_of_words_in_testset**

### `TrigramLanguageModel()`

This class implements a trigram model. Upon initilization, this class first initilizes the `BigramLanguageModel()` and then builds a frequency dictionary for all the bigrams in the train dataset.

#### Probability Calculation

The probability for a token in the trigram model is calculated as follows: 

**count(hv) / count(h)** where, h is the history (the previous 2 words) and v is the current word.

#### Perplexity Calculation

The perplexity for a token in the trigram model is calculated as follows:

**-1 x sum(log(probability_of_all_trigrams_in_testset)) / number_of_words_in_testset**

## Usage / Running the code

The file assumes the following directory structure and files to be present:

```
├── n-gram-models.py                # The source code file
├── 1b_benchmark.train.tokens       # The training dataset
├── 1b_benchmark.dev.tokens         # The validation dataset
└── 1b_benchmark.test.tokens        # The test dataset
```

The model can be trained and tested using the following command:
```
python3 n-gram-models.py
```

The completion of the model will create an `output.txt` file that will contain the information about the models and well as perplexity scores for model evaluation.
