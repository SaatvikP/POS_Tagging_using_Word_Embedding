# POS_Tagging_using_Word_Embedding

## Project Overview

This project implements a Part-of-Speech (POS) tagging model using machine learning techniques. The dataset used is sourced from the Hugging Face batterydata/pos_tagging repository. The approach includes text feature extraction using TF-IDF vectorization and additional handcrafted features, followed by classification using Logistic Regression and Support Vector Machine (SVM) models.

## Dataset

The dataset is loaded from Hugging Face and consists of:
Train Set: 13,054 samples (We use 1000 samples for now)
Test Set: 1,451 samples

Each sample contains:
words: A list of words forming a sentence.
labels: The corresponding POS tags for each word.

## Feature Extraction
The following features are extracted for training:
TF-IDF Vectorization: Character-level n-grams (bi-grams and tri-grams) are extracted using TfidfVectorizer.
### Extra Features:
Whether the word starts with an uppercase letter.
Whether the word is a punctuation mark.

## Model Training

Two models are trained on the extracted features:
### Logistic Regression:
Configured with max_iter=5000 for convergence.
Achieved an accuracy of 75.03% on the test set.

### Support Vector Machine (SVM):
Configured with an RBF kernel (kernel='rbf') and default hyperparameters.
Achieved an accuracy of 77.67% on the test set.
