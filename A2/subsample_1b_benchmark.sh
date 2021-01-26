#!/usr/bin/env bash

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
cd 1-billion-word-language-modeling-benchmark-r13output/
cd heldout-monolingual.tokenized.shuffled
echo "Creating train set"
cat news.en.heldout-0000*-of-00050 > ../../1b_benchmark.train.tokens

# Create the dev set
echo "Creating dev set"
cat news.en.heldout-00010-of-00050 news.en.heldout-00011-of-00050 > ../../1b_benchmark.dev.tokens

# Create the test set
echo "Creating test set"
cat news.en.heldout-00012-of-00050 news.en.heldout-00013-of-00050 > ../../1b_benchmark.test.tokens
