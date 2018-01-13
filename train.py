import tensorflow as tf
import numpy as np
import pickle
import keras
import argparse
from tqdm import tqdm
from keras.optimizers import RMSprop
from model import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

keras.backend.set_session(sess)

epochs = 200
batch_size = 50
seqence_len = 58
num_samples = 1000

def main():
    training_data = []
    with open('pinyin.txt', 'r') as input_file:
        raw_data = input_file.readlines()
        for line in raw_data:
            line = line.strip()
            training_data.append(line.split(' '))
    #import ipdb; ipdb.set_trace()
    # one hot encoding
    word2idx = pickle.load(open('word2idx.p', 'rb'))
    vocab_size = len(word2idx)
    print(vocab_size)
    batch_X = []
    #import ipdb; ipdb.set_trace()
    #for line in tqdm(training_data[:num_samples]):
    #    sequence_vector = []
    #    for word in line:
    #        bi = np.zeros(vocab_size)
    #        if word in word2idx.keys():
    #            index = word2idx[word]
    #        else:
    #            index = word2idx[' ']
    #        bi[index] = 1
    #        sequence_vector.append(bi)
    #    batch_X.append(sequence_vector)
    for line in tqdm(training_data):
        sequence_vector = []
        for word in line:
            if word in word2idx.keys():
                idx = word2idx[word]
            else:
                idx = word2idx[' ']
            sequence_vector.append(idx)
        batch_X.append(sequence_vector)
    batch_Y = []
    print("make batch_Y")
    for i in tqdm(range(len(batch_X) - 1)):
        batch_Y.append(batch_X[i+1])
    batch_Y.append(batch_X[0]) # QQ
    # build model
    model = Autoencoder(256) # latent_dim
    model.build(seqence_len, vocab_size)
    print("build model done")
    model.autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    print(model.autoencoder.summary())
    model.autoencoder.fit(batch_X, batch_Y, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()

