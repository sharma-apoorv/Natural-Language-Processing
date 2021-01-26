# Classifiers

## Description

This contains 1 python file `classifiers.py`. The file has 2 main sentiment classifiers:

1. Lexicon based classifier
2. Logistic regression classifier

## Usage

For the file to be runnable, the following command should be used:
```
python3 classifiers.py
```

The file assumes the following directory structure:

```
├── opinion-lexicon-English      # The lexicon files
│   ├── negative-words.txt       # Negative lexicons
│   └── positive-words.txt       # Positive lexicons
├── txt_sentoken                 # The reviews (files to classify)
│   ├── pos                      # Folder for positive files
│   │   └── cv*_*.txt            # Name does not matter, but they should be `.txt` files
│   └── neg                      # Folder for negative files
│       └── cv*_*.txt            # Name does not matter, but they should be `.txt` files
└── classifiers.py               # Main source code file to run for classification
```

## Dependencies
1. `Python 3.6`
2. Libraries Used:
    * `numpy`