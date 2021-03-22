# Natural-Language-Processing

This repo contains various algorithms from the NLP domain. The following folders contains the following algorithms:

1. A1: Sentiment lexicon-based classifier
	* Code to train a (binary) logistic regression classifier to classify movie reviews as positive or negative
	* The classifier was implemented from scratch, without using any existing implementation of logistic regression, stochastic gradient descent, or automatic differentiation
2. A2: n-gram Modelling
	* Built and evaluated a unigram, bigram and trigram language models
	* The models were evaluated using perplexity scores
	* The code also contains linear interpolation smoothing for better performance of the language models
3. A3: Text classification using GloVE word embeddings
	* Classified movie reviews using pre-trained word embeddings
	* Fine tuned the weights of the word embeddings for better performance
4. A4: Viterbi Algorithm
	* Coded a modified version of the viterbi algorithm to decode the sentences, replacing each “masked” character with a character from the bigram model’s vocabulary. 
5. A5: Byte-Pair Encoding
	* Implemented the BPE algorithm from scratch 
