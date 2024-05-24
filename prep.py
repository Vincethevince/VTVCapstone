import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from prep_input import pairs

# Split data into input and target texts
input_texts = []
target_texts = []

for i in range(0, len(pairs) - 1, 2):
    input_texts.append(pairs[i][0])
    target_texts.append("startseq " + pairs[i][1] + " endseq")

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Determine maximum sequence length
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in target_sequences])

# Pad the sequences
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')

# Create target data shifted by one timestep
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# One-hot encode the target data
num_encoder_tokens = len(tokenizer.word_index) + 1
num_decoder_tokens = len(tokenizer.word_index) + 1

decoder_input_data = to_categorical(decoder_input_data, num_classes=num_decoder_tokens)
decoder_target_data = to_categorical(decoder_target_data, num_classes=num_decoder_tokens)