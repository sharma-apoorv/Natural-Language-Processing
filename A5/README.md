# Byte Pair Encoding

## Introduction

Byte-pair encoding (BPE) is currently a popular technique to deal with this issue. The idea is to encode text using a set of automatically constructed types, instead of conventional word types. A type can be a character or a subword unit; and the types are built through a iterative process, which we now walk you through.

## Description

This assignment requires to implement BPE algorithm. The following steps are followed when encoding a corpus:

Suppose we have the following tiny training data:

*it unit unites*

1. Append a special *<s>* symbol marking the ned of the words
    * This now becomes: *i t <s> u n i t <s> u n i t e s <s>*
2. In each iteration we do the following:
    1. Find the most frequent type of bigram
    2. Merge this into a new symbol
    3. Added it to the type vocabulary

From the examples we would have the following:

1. Bigram to merge: i t
    * Training data: it <s> u n it <s> u n it e s <s>
2. Bigram to merge: it <s>
    * Training data: it<s> u n it<s> u n it e s <s>
3. Bigram to merge: u n
    * Training data: it<s> un it<s> un it e s <s>

In this example, we end up with the type vocabulary:  {i, t, u, n, e, s, <s>, it, it<s>, un}

## Usage / Running the code

The source code assumes the following directory structure and files to be present:

```
├── encoder.py          # The source code file containing the BPE encoder
└── A5-data.txt         # The corpus
```

The simple command will run the code:

```
python3 encoder.py
```

The file produces multiple files, however 2 are important:

1. frequency_one_bpe_scatterplot.png
    * A plot of the vocab size vs the length of the types vocab
2. frequency_one.out
    * Information about the model