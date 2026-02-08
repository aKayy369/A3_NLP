
# Assignment A3: Make Your Own Machine Translation Language  
## AT82.05 – Artificial Intelligence: Natural Language Understanding (NLU)

## Overview

This assignment explores **Neural Machine Translation (NMT)** using a **sequence-to-sequence (Seq2Seq)** architecture with attention mechanisms. The goal is to translate between **English and Nepali**, evaluate different attention mechanisms, and deploy the best-performing model in a simple web application.

The assignment focuses on:

- Dataset preparation and preprocessing  
- Implementing and comparing attention mechanisms  
- Evaluating model performance and interpretability  
- Deploying a trained translation model as a web application  

---


Task 1: Dataset Selection & Preparation :
1. Dataset

Dataset Name: English–Nepali Parallel Corpus

Source: HuggingFace Datasets

Link: https://huggingface.co/datasets/CohleM/english-to-nepali

Number of Sentence Pairs: ~177,000 (subsampled for training efficiency)

License: As provided by the dataset author on HuggingFace

Justification:
This dataset provides aligned English–Nepali sentence pairs from a reputable public repository and is suitable for supervised neural machine translation.

2. Data Preprocessing

The following preprocessing steps were applied:

Text Normalization

Lowercasing English text

Removing extra whitespace

Unicode-safe handling for Nepali text

Tokenization

English: whitespace-based tokenization

Nepali: whitespace tokenization (sufficient for this dataset)

Special Tokens

PAD : padding

SOS : start of sentence

EOS : end of sentence

UNK : unknown token

Vocabulary Construction : 

Separate vocabularies for source (English) and target (Nepali)

Token-to-index and index-to-token mappings created

Padding & Truncation

All sequences padded/truncated to a fixed maximum length

Libraries Used

datasets (HuggingFace)

torch

re (text normalization)

Proper attribution is provided to all dataset and library authors.

Task 2: Experiment with Attention Mechanisms

A Seq2Seq neural network with an LSTM encoder and decoder was implemented. Two attention mechanisms were evaluated.

1. General Attention
𝑒
𝑖
=
𝑠
𝑇
ℎ
𝑖
e
i
	​

=s
T
h
i
	​


Uses dot-product similarity between decoder state and encoder states

Computationally efficient

Requires equal dimensionality

2. Additive (Bahdanau) Attention
𝑒
𝑖
=
𝑣
𝑇
tanh
⁡
(
𝑊
1
ℎ
𝑖
+
𝑊
2
𝑠
)
e
i
	​

=v
T
tanh(W
1
	​

h
i
	​

+W
2
	​

s)

Uses learnable parameters

More expressive due to non-linear transformation

Better suited for complex alignments

Both mechanisms were implemented and trained under identical conditions.

Task 3: Evaluation & Verification
1. Performance Comparison

The models were evaluated using:

Training Loss

Validation Loss

Perplexity (PPL)

Training time (qualitative comparison)

Attention Type	Training Loss	Training PPL	Validation Loss	Validation PPL
General Attention	7.2628	1426.27	8.2373	3779.43
Additive Attention	7.1269	1245.03	8.0299	3071.54
2. Loss Curves

Training and validation loss curves were plotted for both attention mechanisms.

<p align="center"> <img src="images/Learning_curve_Additive.png"> <img src="images/Learning_curve_General.png"> </p>
3. Attention Maps

Attention maps were visualized as heatmaps showing alignment between source and target tokens during translation.

<p align="center"> <img src="images/Attention_map.png"> </p>

These maps provide interpretability by highlighting which source words the model focuses on while generating each target word.

4. Analysis

Additive Attention consistently achieved lower validation loss and perplexity than General Attention, indicating better translation quality and generalization. Although General Attention is computationally cheaper due to its dot-product formulation, it is less expressive. Additive Attention’s learnable parameters and non-linear combination enable more accurate alignment, which is particularly important for English–Nepali translation due to Nepali’s morphological richness and flexible word order. Attention heatmaps further confirm clearer and more focused alignments in the Additive model. Therefore, Additive Attention was selected for deployment.

Task 4: Web Application Development

A web application was developed using Dash to demonstrate real-time English–Nepali machine translation.

Application Features

Text input box for English sentences

Translate button to trigger inference

Display of generated Nepali translation

Model–App Interface Documentation

The web application loads the trained Seq2Seq model with Additive Attention and the saved vocabularies at startup. User input is preprocessed, tokenized, numericalized, and passed through the encoder–decoder model for inference. The predicted Nepali tokens are detokenized and displayed on the interface. The model is loaded once to ensure efficient runtime performance.

App Interface
<p align="center"> <img src="images/image.png"> </p>

Running the App :

cd app
python app.py

