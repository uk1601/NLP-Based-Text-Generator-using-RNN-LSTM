
# NLP based Text Generator using RNN LSTM

<div style="text-align: right">Uday Kiran Dasari</div>

## Objective

The goal of this project is to develop a Telugu language chatbot leveraging deep learning techniques, specifically Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units. The chatbot is designed to understand and generate coherent Telugu text, enhancing natural language processing (NLP) capabilities in the Telugu language.

## Table of Contents

- [Objective](#objective)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Tokenizer Initialization, Sequence Creation, and Padding](#tokenizer-initialization-sequence-creation-and-padding)
- [Model Training](#model-training)
- [Model Evaluation and Saving](#model-evaluation-and-saving)
- [Generating Output](#generating-output)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Contributors](#contributors)
- [References](#references)

## Project Overview

This project involves several key steps to achieve the goal of creating a functional Telugu language chatbot:

1. **Text Preprocessing**
2. **Tokenizer Initialization, Sequence Creation, and Padding**
3. **Preparing Data for Model Training**
4. **Loading the Embeddings**
5. **Model Creation, Training, and Saving**
6. **Loading the Saved Model and Generating Output**

## Dataset

The dataset used in this project consists of Telugu books text data stored in a CSV file. The relevant text data is extracted and preprocessed to create a clean and usable corpus for training the model.

## Preprocessing

The preprocessing steps involve:
- Cleaning and preparing the Telugu text data by removing unwanted characters, normalizing the text, and tokenizing it into words or subwords.
- Merging text data from all rows into a single corpus.
- Splitting the corpus into sentences for further processing.

```python
import re
import pandas as pd

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_and_preprocess_csv(file_path, column_index):
    data = pd.read_csv(file_path)
    corpus = ' '.join(data.iloc[:, column_index].astype(str).tolist())
    preprocessed_corpus = preprocess_text(corpus)
    return preprocessed_corpus

file_path = './Data/telugu_books.csv'
column_index = 1
corpus = read_and_preprocess_csv(file_path, column_index)
print(corpus[:1000])
```

## Tokenizer Initialization, Sequence Creation, and Padding

- Initialized a tokenizer to convert text into numerical sequences.
- Created sequences of fixed length for input into the model.
- Applied padding to ensure uniform sequence lengths, making the data suitable for model training.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

corpus = corpus[:500000]
training_data = [sentence.strip() for sentence in corpus.split('.') if sentence.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_data)
sequences = tokenizer.texts_to_sequences(training_data)
vocab_size = len(tokenizer.word_index) + 1

max_sequence_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
```

## Model Training

Designed and built RNN models with LSTM cells using TensorFlow. The models were trained on the prepared data, adjusting hyperparameters for optimal performance.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, epochs=20, validation_split=0.2)
```

## Model Evaluation and Saving

Evaluated the trained model's performance and saved the model for future use.

```python
model.save('telugu_chatbot_model.h5')
```

## Generating Output

Loaded the saved model to generate responses using a given prompt and produced coherent, contextually appropriate output.

```python
from tensorflow.keras.models import load_model

model = load_model('telugu_chatbot_model.h5')

def generate_text(model, tokenizer, input_text, max_sequence_length):
    # Implementation for generating text based on input_text
    pass
```

## Conclusion

The project successfully developed a Telugu language chatbot capable of generating coherent text using RNN with LSTM units. The chatbot leverages deep learning to enhance NLP capabilities in the Telugu language.

## Future Work

- Improve the model's accuracy and efficiency.
- Expand the dataset for more diverse training data.
- Integrate the chatbot into a user-friendly application.

## Dependencies

- TensorFlow
- Pandas
- NumPy

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Run the Jupyter notebook `NLP Telugu.ipynb`.

## Contributors

- Uday Kiran Dasari

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
