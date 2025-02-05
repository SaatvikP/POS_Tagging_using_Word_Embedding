# POS_Tagging_using_Word_Embedding

## Project Overview
This project implements Part-of-Speech (POS) tagging using both machine learning and deep learning techniques. The dataset used is sourced from the Hugging Face batterydata/pos_tagging repository.
The project  includes:
Traditional Machine Learning Models: Logistic Regression and Support Vector Machine (SVM) using TF-IDF feature extraction.
Deep Learning Model: Bidirectional LSTM (BiLSTM) trained with pre-trained GloVe embeddings.

## Dataset

The dataset is loaded from Hugging Face and consists of:
Train Set: 13,054 samples (We use 1000 samples for now)
Test Set: 1,451 samples

Each sample contains:
words: A list of words forming a sentence.
labels: The corresponding POS tags for each word.
## Approach 1: Traditional Machine Learning
We extract handcrafted features and use them for Logistic Regression and SVM classifiers.
### Feature Extraction
The following features are extracted for training:
TF-IDF Vectorization: Character-level n-grams (bi-grams and tri-grams) are extracted using TfidfVectorizer.
#### Extra Features:
Whether the word starts with an uppercase letter.
Whether the word is a punctuation mark.

### Model Training

Two models are trained on the extracted features:
#### Logistic Regression:
Configured with max_iter=5000 for convergence.
Achieved an accuracy of 75.03% on the test set.

#### Support Vector Machine (SVM):
Configured with an RBF kernel (kernel='rbf') and default hyperparameters.
Achieved an accuracy of 77.67% on the test set.

## Approach 2: Deep Learning with BiLSTM
We use a Bidirectional LSTM (BiLSTM) model trained on word embeddings.
### Model Architecture
Embedding Layer: Initialized with pre-trained GloVe embeddings (glove-wiki-gigaword-100).
Bidirectional LSTM: 64 hidden units with recurrent_dropout=0.1.
TimeDistributed Dense Layer: Predicts POS tags at each timestep.
### Training & Results
Trained for 3 epochs with a batch size of 32.
Optimizer: Adam
Loss Function: Categorical Crossentropy
Accuracy on the test set: 94.78%

## Files in the Repository
POS_Tagging_SVM_LogReg.py → Implements POS tagging using Logistic Regression and SVM with TF-IDF feature extraction.
POS Tagging_BiLSTM.py → Implements POS tagging using BiLSTM with GloVe embeddings.
