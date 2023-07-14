# Quran Neural Parser

*Join us on a new journey! Visit the [Corpus 2.0 upgrade project](https://github.com/kaisdukes/quranic-corpus) for new work on the Quranic Arabic Corpus.*

## What’s in this Repo?

A neural parser for the [Quranic Treebank](https://qurancorpus.app/treebank/2:258). Parses the Quran into dependency graphs based on traditional *i’rāb* (إعراب), using neural networks.

To work with this codebase, you will need a strong background in Artificial Intelligence applied to Quranic Research, specifically in the fields of Computational Linguistics and Natural Language Processing (NLP).

## Why Do We Need This?

The Quranic Arabic Corpus provides detailed morphological analysis for the entire Quran. However, the Quranic Treebank, which provides syntactic analysis using *i’rāb* (إعراب), is not yet complete. The ability to parse and understand the Quran from a linguistic perspective is crucial to Quranic Research. The task of completing the Quranic Treebank is ambitious. Applying AI to the process will significantly speed up the work and lead to more accurate results, in combination with expert human review.

The aim of this repository is to provide a baseline model for experimenting with more advanced neural parsing methods to aid completion of the treebank. Related Quranic AI work for the corpus includes:

* [Quran Neural Chunker](https://github.com/kaisdukes/quran-neural-chunker): A data preprocessor for the Quranic Treebank using neural networks. Divides longer verses into smaller chunks.
* [Quran SVM Parser](https://github.com/kaisdukes/quran-svm-parser): The original Quranic Arabic Corpus parser using SVM-based machine learning, from Dukes & Habash's 2011 paper.

## Please Use Official APIs and Avoid Distributing Draft Data

The Quranic Arabic Corpus provides comprehensive linguistic annotation for each word in the Quran based on authentic traditional sources, while respecting its deep cultural significance. Linguistic analysis of the Quran requires expert knowledge and is subject to continual refinement and improvement. The code in this repository uses official Corpus APIs to locally cache the Quranic Treebank. Given the constant refinement by our Linguistic Team and our partner universities, we kindly ask to avoid redistributing this draft data. Please promote the use of official APIs to ensure users always have the most up-to-date and accurate data available.

## Getting Started

This project uses [Poetry](https://python-poetry.org) to manage package dependencies.

First, clone the repository:

```
git clone https://github.com/kaisdukes/quran-neural-parser.git
cd quran-neural-parser
```

Install Poetry using [Homebrew](https://brew.sh):

```
brew install poetry
```

Next, install project dependencies:

```
poetry install
```

All dependencies, such as [pandas](https://pandas.pydata.org), are installed in the virtual environment.

Use the Poetry shell:

```
poetry shell
```

Test the parser:

```
python tests/parser_test.py
```