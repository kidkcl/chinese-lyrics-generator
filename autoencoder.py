from keras.layers import Input, LSTM, RepeatVector, Embedding
from keras.models import Model

class Autoencoder:
     
    def __init__(self, latent_dim=256):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.latent_dim = latent_dim
        print(self.latent_dim)

    def build(self, sequence_length, input_dim):
        inputs = Input(shape=(sequence_length, input_dim))
        encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(sequence_length)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        encoded_input = Input(shape=(None,self.latent_dim))
        decoded_layer = sequence_autoencoder.layers[-1]
        decoder = Model(encoded_input, decoded_layer(encoded_input))

        self.autoencoder = sequence_autoencoder
        self.encoder = encoder
        self.decoder = decoder

