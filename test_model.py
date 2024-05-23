from preprocessing import input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, target_docs, input_tokens, target_tokens, max_encoder_seq_length, num_encoder_tokens, num_decoder_tokens
from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras.models import Model, load_model
import numpy as np
import re
import tensorflow as tf
from training_model import decoder_inputs, decoder_lstm, decoder_dense

latent_dim = 256

from tensorflow.keras.utils import get_custom_objects

# Definiere die benutzerdefinierte Schicht (NotEqual)
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        return tf.math.not_equal(x, y)

class Any(tf.keras.layers.Layer):
    def __init__(self, axis=None, keepdims=False, **kwargs):
        super(Any, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_any(inputs, axis=self.axis, keepdims=self.keepdims)

# Registriere die benutzerdefinierten Schichten
get_custom_objects().update({'NotEqual': NotEqual, 'Any': Any})

# Laden des gesamten Modells
#training_model = load_model('training_model.h5', custom_objects={'Masking': Masking, 'NotEqual': NotEqual, 'Any': Any})
training_model = load_model('training_model.h5')

# Ausgabe der Schichten des Modells
for i, layer in enumerate(training_model.layers):
    print(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")

# Define the encoder model
#encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_inputs = training_model.input[0]
#encoder_mask = Masking(mask_value=0)(encoder_inputs)
#encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_lstm = training_model.layers[2]
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Define the decoder model
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

#decoder_inputs = Input(shape=(None, num_decoder_tokens))
#decoder_mask = Masking(mask_value=0)(decoder_inputs)
#decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

#decoder_inputs = training_model.input[1]
#decoder_lstm = training_model.layers[3]
#decoder_dense = training_model.layers[4]
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
#decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [decoder_state_hidden, decoder_state_cell]

# Create the training model
#training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#training_model.load_weights("training_model.h5")

# Create the inference models
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + [decoder_state_hidden, decoder_state_cell])
encoder_model = Model(encoder_inputs, encoder_states)

def decode_sequence(test_input):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(test_input)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first token of target sequence with the start token.
    if '<START>' in target_features_dict:
        target_seq[0, 0, target_features_dict['<START>']] = 1.
    else:
        raise KeyError("'<START>' token not found in target_features_dict")

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        # Run the decoder model to get possible 
        # output tokens (with probabilities) & states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

        # Choose token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # Exit condition: either hit max length
        # or find stop token.
        if (sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [hidden_state, cell_state]

    return decoded_sentence.replace("<START>", "").replace("<END>", "").strip()

def decode_sequence_beam_search(input_seq, beam_width=3):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first token of target sequence with the start token.
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # Initialize variables
    sequences = [[list(), 0.0, states_value]]  # [seq, score, state]

    # Loop until max length
    for _ in range(max_decoder_seq_length):
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score, states = sequences[i]
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            if seq:
                target_seq[0, 0, seq[-1]] = 1.
            output_tokens, h, c = decoder_model.predict([target_seq] + states)
            states_value = [h, c]

            # Get probabilities for the top beam_width tokens
            top_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]

            for index in top_indices:
                candidate = [seq + [index], score - np.log(output_tokens[0, -1, index]), states_value]
                all_candidates.append(candidate)

        # Sort candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # Select top beam_width
        sequences = ordered[:beam_width]

        # Check if any sequence is finished
        for seq, score, states in sequences:
            if seq[-1] == target_features_dict['<END>']:
                return ' '.join(reverse_target_features_dict[index] for index in seq)

    # Return the best sequence if no <END> token was found
    return ' '.join(reverse_target_features_dict[index] for index, score, states in sequences[0])

