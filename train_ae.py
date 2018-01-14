import tensorflow as tf
import numpy as np
import pickle
import keras
from gensim.models import word2vec
from tqdm import tqdm
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from autoencoder import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

keras.backend.set_session(sess)

epochs = 200
batch_size = 50
sequence_len = 39
num_samples = 1000
input_dim = 100

def main():
    training_data = []
    with open('pinyin.txt', 'r') as input_file:
        raw_data = input_file.readlines()
        for line in raw_data:
            line = line.strip()
            training_data.append(line.split(' '))
    # load pre-trained word2vec model
    wv_model = word2vec.Word2Vec.load('pretrain100.model.bin')
    batch_X = []
    batch_Y = []
    for i in tqdm(range(len(training_data))):
        sequence_vec = []
        for word in training_data[i]:
            sequence_vec.append(wv_model.wv[word])
        batch_X.append(sequence_vec)
        if i is not 0:
            batch_Y.append(sequence_vec)
    batch_Y.append(batch_X[-1])
    encoder_inputs = np.asarray(batch_X)
    decoder_outputs = np.asarray(batch_Y)
    print(encoder_inputs.shape)
    print(decoder_outputs.shape)
    # build model
    model = Autoencoder(256) # latent_dim
    model.build(sequence_len, input_dim)
    print("build model done")
    model.autoencoder.compile(optimizer='rmsprop', loss='mse')
    model.autoencoder.summary()
    filepath = 'test_dim256/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    model.autoencoder.fit(encoder_inputs, decoder_outputs, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    model.autoencoder.save('seq_autoencoder.h5')

if __name__ == "__main__":
    main()

