# NLP based Text Generator using RNN LSTM


## Objective

The goal of this project is to develop a Telugu language chatbot leveraging deep learning techniques, specifically Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units. The chatbot is designed to understand and generate coherent Telugu text, enhancing natural language processing (NLP) capabilities in the Telugu language.

## Table of Contents

- [Objective](#objective)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Tokenizer Initialization, Sequence Creation, and Padding](#tokenizer-initialization-sequence-creation-and-padding)
- [Loading the embedding and creation of the embedding matrix](#Loading-the-embedding-and-creation-of-the-embedding-matrix)
- [Model Creation, Training, and Saving](#Model-Creation,-Training,-and-Saving)
- [Generating Output](#generating-output)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Project Overview

This project involves several key steps to achieve the goal of creating a functional Telugu language chatbot:

1. **Text Preprocessing**
2. **Tokenizer Initialization, Sequence Creation, and Padding**
3. **Preparing Data for Model Training**
4. **Loading the embedding and creation of the embedding matrix**
5. **Model Creation, Training, and Saving**
6. **Loading the Saved Model and Generating Output**

## Dataset

The dataset used in this project consists of Telugu books text data stored in a CSV file [Data](https://www.kaggle.com/datasets/sudalairajkumar/telugu-nlp). The relevant text data is extracted and preprocessed to create a clean and usable corpus for training the model.

## Preprocessing

The preprocessing steps involve:
- Cleaning and preparing the Telugu text data by removing unwanted characters, normalizing the text, and tokenizing it into words or subwords.
- Merging text data from all rows into a single corpus.
- Splitting the corpus into sentences for further processing.


## Tokenizer Initialization, Sequence Creation, and Padding

- Initialized a tokenizer to convert text into numerical sequences.
- Created sequences of fixed length for input into the model.
- Applied padding to ensure uniform sequence lengths, making the data suitable for model training.

## Loading the embedding and creation of the embedding matrix

- FastText Telugu embeddings are used. [Telugu Embeddings](https://fasttext.cc/docs/en/crawl-vectors.html)
- Loaded pre-trained word embeddings to represent the words in a dense vector space.
- Integrated these embeddings into the model to enhance its understanding of the language context.

## Model Creation, Training, and Saving

- Designed and built 2 RNN models with LSTM cells using TensorFlow.
- Trained the model on the prepared data, adjusting hyperparameters for optimal performance.
- Saved the trained model for future use.

## Generating Output

- Loaded the saved model for generating responses using two approaches **Deterministic and Probabilistic**
- Provided a prompt to the model and generated coherent, contextually appropriate output.

## Conclusion

This project successfully demonstrates the creation of a Telugu language chatbot using RNNs with LSTM units. By leveraging pre-trained word embeddings and training two different LSTM architectures, the model is capable of generating coherent Telugu text. The generated text highlights the model's ability to understand and predict language structures effectively, paving the way for advanced NLP applications in Telugu.

However, it is important to note that the model's performance is constrained by the limited data and type of data it was trained on. As a result, the range of emotions and vocabulary is also limited. Consequently, the model's predictions may sometimes loop within the restricted vocabulary provided. Expanding the dataset with more diverse and extensive text could further enhance the model's capabilities and robustness.

## Future Work

- Try different model architectures.
- Improve the model's accuracy and efficiency by incorporating techniques such as Dropout layers, regularization layers, Callbacks, and optimizers.
- Expand the dataset for more diverse training data.
- Integrate the chatbot into a user-friendly application.

## Dependencies

- TensorFlow
- Pandas
- NumPy

## How to Run

1. Clone the repository.
2. Install the required dependencies and download the required files.
3. Run the Jupyter notebook `NLP Telugu.ipynb`.
