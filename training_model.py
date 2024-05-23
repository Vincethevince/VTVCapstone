from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data, max_encoder_seq_length, max_decoder_seq_length

# Add Dense to the imported layers
from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Choose dimensionality
dimensionality = 256

# Choose the batch size
# and number of epochs:
batch_size = 32
epochs = 500

# Encoder training setup
encoder_inputs = Input(shape=(None, num_encoder_tokens))
#encoder_mask = Masking(mask_value=0)(encoder_inputs)
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder training setup:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
#decoder_mask = Masking(mask_value=0)(decoder_inputs)
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Building the training model:
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model:
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model:
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

training_model.save('training_model.h5')
