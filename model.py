from keras.layers import Input, LSTM, RepeatVector, Embedding
from keras.models import Model

class Autoencoder:
     
    def __init__(self, latent_dim=64):
        self.encoder = None
        self.autoencoder = None
        self.latent_dim = latent_dim
        print(self.latent_dim)

    def build(self, sequence_length, input_dim):
        inputs = Input(shape=(None,))
        embedded = Embedding(input_dim, 100, input_length=sequence_length)(inputs)
        encoded = LSTM(self.latent_dim)(embedded)

        decoded = RepeatVector(sequence_length)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        self.autoencoder = sequence_autoencoder
        self.encoder = encoder
        return sequence_autoencoder, encoder


