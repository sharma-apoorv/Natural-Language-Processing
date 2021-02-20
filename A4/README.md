# Conditional Random Fields

## Introduction

The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events

This assignment requires to modify the Viterbi algorithm to find the best path through a series of masked characters in a sentence. 

## Description

Consider the following sentence:

```
<start> x1 x2 <mask> <mask> x3 <eos>
```

Each token here represents a character. The `<mask>` tokens represent unknown words, where as `x*` represent known characters. 

The goal is to find the best character (most likely character), using a pre-specified language model. A potential language model may look as follows:

```
<start> <start>	0.11
<start> x1 	0.21
<start> x2	0.31
<start> x3	0.41
<start> <eos>  	0.51
x1 <start> 	0.12
x1 x1	0.22
x1 x2 	0.32
x1 x3 	0.42
x1 <eos> 	0.52
x2 <start> 	0.13
x2 x1	0.23
x2 x2 	0.33
x2 x3 	0.43
x2 <eos> 	0.53
x3 <start> 	0.14
x3 x1	0.24
x3 x2 	0.34
x3 x3 	0.44
x3 <eos> 	0.54
<eos> <start> 	0.15
<eos> x1	0.25
<eos> x2 	0.35
<eos> x3 	0.45
<eos> <eos> 	0.55
```

For example, for the case of `<start> x1 	0.21`, this means the **p(x1 | <start>) = 0.21**. 

## Usage / Running the code

The source code assumes the following directory structure and files to be present:

```
├── crf.py                  # The source code file containing the modified Viterbi algorithm
├── lm.txt                  # The language model specified
└── 15pctmasked.txt         # The sentences to 'decode'
```

The simple command will run the code:

```
python3 crf.py
```

This will produce a decode file called `unmasked.txt` in the same directory as the source code file. This file will contain sentence that have the `<mask>` characters replaced with the most likely characters. 


A full detail of the potential arguments have been specified below: 

```
usage: crf.py [-h] [-lm LANG_MODEL_PATH] [-ip INPUT_FILE_PATH]
              [-op OUTPUT_FILE_PATH] [-t]

Viterbi Algorithm

optional arguments:
  -h, --help            show this help message and exit
  -lm LANG_MODEL_PATH, --lang-model LANG_MODEL_PATH
                        This is the path to the language model file
  -ip INPUT_FILE_PATH, --input-file INPUT_FILE_PATH
                        This is the path to the file that contains the masked
                        sentences
  -op OUTPUT_FILE_PATH, --output-file OUTPUT_FILE_PATH
                        This is the path to file that will be output
  -t, --sanity-check    Flag to indicate whether to perform sanity checking
                        or not
```