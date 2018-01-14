from keras.layers import Input, LSTM, RepeatVector, Embedding, Concatenate, Dense
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

        encoder_outputs = RepeatVector(sequence_length)(encoded)
        decoder_inputs = Input(shape=(None,))
        decoder_embedded = Embedding(input_dim, 100, input_length=sequence_length)(decoder_inputs)
        decoder_concat = Concatenate(axis=2)([encoder_outputs, decoder_embedded])
        decoded = LSTM(self.latent_dim, return_sequences=True)(decoder_concat)
        decoder_outputs = Dense(input_dim, activation='softmax')(decoded)

        sequence_autoencoder = Model([inputs, decoder_inputs], decoder_outputs)
        encoder = Model(inputs, encoded)
        self.autoencoder = sequence_autoencoder
        self.encoder = encoder
        return sequence_autoencoder, encoder


