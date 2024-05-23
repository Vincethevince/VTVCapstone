import numpy as np
import re
from test_model import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length, decode_sequence_beam_search

class ChatBot:
  
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
  
    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on dialog. Would you like to chat with me?\n")
        if user_response.lower() in self.negative_responses:
            print("Ok, have a great day!")
            return
        self.chat(user_response)
  
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply))
  
    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    
        for timestep, token in enumerate(tokens[:max_encoder_seq_length]):  # Truncate tokens if they exceed max length
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
            else:
                print(f"Warning: Token '{token}' not in input features dictionary.")
        return user_input_matrix
  
    def generate_response(self, user_input):
        user_input_matrix = self.string_to_matrix(user_input)
        states_value = encoder_model.predict(user_input_matrix)

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        if '<START>' in target_features_dict:
            target_seq[0, 0, target_features_dict['<START>']] = 1.
        else:
            raise KeyError("'<START>' token not found in target_features_dict")

        decoded_sentence = ''
        stop_condition = False
        while not stop_condition:
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict.get(sampled_token_index, '<UNK>')
            decoded_sentence += " " + sampled_token

            if sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [hidden_state, cell_state]

        return decoded_sentence.replace("<START>", "").replace("<END>", "").strip()
  
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply.lower():
                print("Ok, have a great day!")
                return True
        return False

chatty_mcchatface = ChatBot()
chatty_mcchatface.start_chat()
