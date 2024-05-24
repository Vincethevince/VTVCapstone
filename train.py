from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import re
from tensorflow.keras.callbacks import EarlyStopping
from prep import max_encoder_seq_length, num_encoder_tokens, num_decoder_tokens, encoder_input_data,decoder_input_data,decoder_target_data, tokenizer,max_decoder_seq_length
# Hyperparameters
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_encoder_seq_length,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim, input_length=max_encoder_seq_length)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=100,
          validation_split=0.2, callbacks=[early_stopping])


encoder_model = Model(encoder_inputs, encoder_states)

# Define the decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first token of target sequence with the start token.
    target_seq[0, 0, tokenizer.word_index['startseq']] = 1.

    # Sampling loop for a batch of sequences
    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length or find stop character.
        if sampled_char == 'endseq' or len(decoded_sentence) > max_decoder_seq_length:
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def string_to_matrix(user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input.lower())
    user_input_matrix = np.zeros((1, max_encoder_seq_length), dtype='int32')
    
    for timestep, token in enumerate(tokens[:max_encoder_seq_length]):
        if token in tokenizer.word_index:
            user_input_matrix[0, timestep] = tokenizer.word_index[token]
    
    return user_input_matrix

# Example usage
test_input = "Hi how are you"
while test_input!= "exit":
    test_input_matrix = string_to_matrix(test_input)
    test_input = input(decode_sequence(test_input_matrix))
