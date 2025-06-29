# Grammar Error Correction with Attention: A Keras Seq2Seq RNN Approach

---

## 📚 Project Overview

Welcome to the **Grammar Error Correction (GEC) with Attention** project! This repository contains code to train a neural network model that can automatically detect and correct grammatical errors in English sentences.

Grammar Error Correction is a challenging task in Natural Language Processing (NLP), aiming to transform an "incorrect" sentence into its "correct" counterpart. It's a sequence-to-sequence problem, much like machine translation, but instead of translating between languages, we're translating between an erroneous sentence and a grammatically correct one in the same language.

This project implements a classic and effective approach: a **Sequence-to-Sequence (Seq2Seq) model built with Recurrent Neural Networks (RNNs)**, specifically LSTMs, and enhanced with an **Attention Mechanism**.

## ✨ Features

*   **Seq2Seq Architecture:** Utilizes an Encoder-Decoder framework.
*   **Bidirectional Encoder:** Processes the input sentence from both directions for better context understanding.
*   **Attention Mechanism:** Allows the decoder to focus on relevant parts of the input sentence while generating each word of the corrected sentence.
*   **Keras/TensorFlow Implementation:** Built using the popular and user-friendly Keras API.
*   **Data Preprocessing Pipeline:** Includes cleaning, tokenization, adding special tokens (`sos`, `eos`), and padding.
*   **Interactive Demo:** Includes a simple Gradio interface to easily test the trained model.

## 🤔 Why Seq2Seq with Attention for GEC?

GEC involves transforming a sequence (incorrect sentence) into another sequence (correct sentence), making it a perfect fit for Seq2Seq models.

*   **RNNs (LSTMs):** Are good at processing sequential data, capturing dependencies between words.
*   **Encoder-Decoder:** The encoder builds a representation of the input sentence, and the decoder uses this representation to generate the output sentence.
*   **Attention:** Traditional Seq2Seq models struggle with long sequences as the fixed-size encoder state becomes a bottleneck. Attention allows the decoder to look back at the entire encoder output sequence at each step, dynamically deciding which input words are most relevant for generating the next output word. This significantly improves performance, especially for longer sentences and complex corrections.

## 📐 Model Architecture

The model follows a standard Seq2Seq architecture with Bahdanau-style global dot product attention:

1.  **Encoder:**
    *   Takes the padded sequence of word IDs as input.
    *   An **Embedding layer** maps word IDs to dense vectors.
    *   A **Bidirectional LSTM** processes the embedded sequence, generating a contextualized representation for each input word (`encoder_outputs`) and summarizing the entire sequence into final hidden and cell states (`encoder_states`).

2.  **Decoder:**
    *   Takes the padded target sequence (shifted right, starting with `<sos>`) as input during training. During inference, it takes one generated token at a time.
    *   An **Embedding layer** (shared with the encoder's embedding layer, or a separate one depending on design - here it seems to be the same vocabulary) maps target word IDs to dense vectors.
    *   An **LSTM** processes the embedded target sequence. Its initial state is set to the final states of the encoder.

3.  **Attention:**
    *   A custom **AttentionLayer** calculates attention scores between the decoder's current hidden state and all of the encoder's output timesteps (`encoder_outputs`).
    *   These scores are converted to attention weights using softmax.
    *   A **context vector** is computed as a weighted sum of the `encoder_outputs` based on the attention weights.

4.  **Output:**
    *   The decoder's LSTM output for the current timestep is **concatenated** with the context vector from the attention mechanism.
    *   A final **Dense layer** with a softmax activation predicts the probability distribution over the entire vocabulary for the next word.

This concatenated approach allows the model to leverage both the sequential decoding context and the dynamically weighted input context when making predictions.

## 📊 Dataset

The project uses a custom CSV dataset (`Total_final_dataset.csv`) containing pairs of incorrect (`input`) and corrected (`target`) sentences.

For training efficiency and feasibility in a standard Colab environment, the notebook samples a subset (e.g., 50,000 examples) from the full dataset. The `target` column, potentially containing multiple comma-separated corrections, is simplified to use only the first provided correction.

## ⚙️ Preprocessing Pipeline

The raw text data goes through the following steps:

1.  **Load and Sample:** Read CSV and sample a subset.
2.  **Format:** Ensure 'incorrect' and 'corrected' columns are available, taking the first correction from the 'target' list.
3.  **Clean Text:**
    *   Standardize whitespace.
    *   Convert to lowercase.
    *   Remove punctuation and digits (a simplification that might remove useful information but reduces vocabulary size).
4.  **Handle Empty Rows:** Remove pairs where either the incorrect or corrected sentence becomes empty after cleaning.
5.  **Add Special Tokens:** Prepend `sos` (start of sequence) and append `eos` (end of sequence) tokens to the *corrected* sentences. These act as signals for the decoder.
6.  **Tokenization:**
    *   A single `tf.keras.preprocessing.text.Tokenizer` is fitted on *all* text (incorrect and corrected, including `sos`/`eos` tokens).
    *   A vocabulary is built, potentially limited by `VOCAB_LIMIT` (e.g., 20,000 words). An `<unk>` token handles out-of-vocabulary words.
    *   Sentences are converted into sequences of integer word IDs.
7.  **Padding:**
    *   Sequences are padded with zeros (`padding='post'`) to a fixed maximum length (`MAX_INPUT_LEN`, `MAX_TARGET_LEN`). This length is determined from the data (e.g., max length + buffer, capped at 128) to ensure uniform input size for the model.

## 🎓 Training

The model is trained using:

*   **Optimizer:** Adam
*   **Loss Function:** Sparse Categorical Crossentropy (suitable for integer targets predicting probabilities over a vocabulary).
*   **Metrics:** Accuracy.
*   **Callbacks:**
    *   `ModelCheckpoint`: Saves the model weights with the best validation accuracy to Google Drive.
    *   `EarlyStopping`: Monitors validation accuracy and stops training if it doesn't improve for a specified number of epochs (`patience`).

During training, the decoder uses **teacher forcing**, where the correct target token from the previous timestep is fed as input at the current timestep, rather than the model's own prediction.

## ➡️ Inference (Prediction)

Generating a corrected sentence for a new input requires a different process than training:

1.  **Separate Inference Models:** Two dedicated Keras `Model` instances are built, reusing the trained layers:
    *   An `encoder_model` that takes the input sequence and returns the final encoder states and the sequence of encoder outputs.
    *   A `decoder_model` that takes a single input token (starting with `<sos>`), the encoder outputs (for attention), and the previous decoder states, and outputs the probability distribution for the next token, the new decoder states, and the attention weights.
2.  **Step-by-Step Decoding:**
    *   The input sentence is cleaned, tokenized, and padded.
    *   The `encoder_model` processes the input to get the initial decoder state and encoder outputs.
    *   The decoder starts with the `<sos>` token as its first input and the encoder's final states.
    *   In a loop, the `decoder_model` predicts the next token based on the current input token, previous states, and encoder outputs (via attention).
    *   The token with the highest probability (greedy decoding) is selected as the predicted word.
    *   This predicted word becomes the input for the next step.
    *   The process continues until the `eos` token is predicted or a maximum output length is reached.

## 🚀 Interactive Demo with Gradio

To make it easy to test the model, a simple web interface is created using the Gradio library. This interface allows users to type an incorrect sentence and get the corrected output from the trained model directly within the Colab notebook or via a public shareable link.

## 🏃 How to Run

This project is designed to be run in a Google Colab environment, especially if you intend to train the model yourself, as it utilizes a GPU.

1.  **Clone the repository:** Clone this GitHub repository to your local machine or directly open the `.ipynb` file in Google Colab.
2.  **Upload Data:** Upload your `Total_final_dataset.csv` file to your Colab environment or Google Drive. Adjust the `data_file_path` variable in Cell 2 if needed.
3.  **Mount Google Drive (Optional but Recommended):** If you want to save/load model weights, mount your Google Drive in Colab using the provided snippet (often the first cell in a new Colab notebook or add a code cell with `from google.colab import drive; drive.mount('/content/drive')`). Update the `save_dir` path in the training cell (Cell 12) accordingly.
4.  **Open the Notebook:** Open the `juriRNN_with_Att_for_NMT.ipynb` notebook in Google Colab.
5.  **Run Cells:**
    *   **To Train:** Run all cells sequentially from top to bottom. Ensure you have a GPU runtime enabled (Runtime -> Change runtime type -> GPU). The training might take a significant amount of time depending on the dataset size and number of epochs.
    *   **To Use Pre-trained Weights:**
        1.  Mount Google Drive and upload the saved weights file (`gec_simplifier_final_model.weights.h5`) to the specified `save_dir` path.
        2.  Run Cell 1 (Imports).
        3.  Run Cells 2 and 3 (Data Loading, Cleaning, Tokenization) - **REQUIRED** to set up the tokenizer and global parameters (`VOCAB_SIZE`, `MAX_INPUT_LEN`, `MAX_TARGET_LEN`) correctly.
        4.  Run Cells 6, 7, 8, 9, 10, 11 (Model Architecture Definition). This defines the *structure* and creates the layer objects.
        5.  **Add a new cell after Cell 11** with the code to load the weights (as shown in the "How to Load Saved Weights" example provided in the explanation response). Execute this cell.
        6.  Run Cell 15 (Build Inference Models). These models now inherit the loaded weights from the shared layers.
        7.  Run Cell 16 (Prediction Function Definition).
        8.  Run Cell 17 (Gradio App).
6.  **Interact:** Once the Gradio cell finishes executing, a public URL will appear. Click on it to open the web interface and start correcting sentences!

## 📂 Files

*   `juriRNN_with_Att_for_NMT.ipynb`: The main colab notebook containing all the code for data loading, preprocessing, model building, training, inference, and the Gradio app.
*   `Total_final_dataset.csv`: The dataset file used for training.
