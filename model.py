from keras.layers import Input, LSTM, RepeatVector, Embedding
from keras.models import Model

class Autoencoder:
     
    def __init__(self, latent_dim=256):
        self.encoder = None
        self.autoencoder = None
        self.latent_dim = latent_dim
        print(self.latent_dim)

    def build(self, sequence_length, input_dim):
        inputs = Input(shape=(sequence_length, input_dim))
        inputs = Embedding(input_dim, 100, input_length=sequence_length)(inputs)
        encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(sequence_length)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        sequence_autoencoder.summary()
        self.autoencoder = sequence_autoencoder
        self.encoder = encoder


