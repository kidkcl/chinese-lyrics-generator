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

epochs = 50
batch_size = 50
seqence_len = 39
num_samples = 1000

def main():
    training_data = []
    with open('test.txt', 'r') as input_file:
        raw_data = input_file.readlines()
        for line in raw_data:
            line = line.strip()
            training_data.append(line.split(' '))
    #import ipdb; ipdb.set_trace()
    # one hot encoding
    word2idx = pickle.load(open('word2idx.p', 'rb'))
    idx2word = pickle.load(open('idx2word.p', 'rb'))
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
    #import ipdb; ipdb.set_trace()
    for line in tqdm(training_data):
        sequence_vector = []
        for word in line:
            if word in word2idx.keys():
                idx = word2idx[word]
            else:
                idx = word2idx[' ']
            sequence_vector.append(idx)
        batch_X.append(sequence_vector)
    encoder_input = np.asarray(batch_X)
    batch_Y = np.zeros(encoder_input.shape)
    print("make batch_Y")
    for i in tqdm(range(len(batch_Y))):
        batch_Y[i][0] = word2idx[' ']
    decoder_input = np.asarray(batch_Y)
    print(encoder_input.shape)
    # build model
    model = keras.models.load_model('s2s_128/weights-improvement-84-0.7450.hdf5')
    #model = keras.models.load_model('s2s_64_100_5000.h5')
    output = []
    for i in range(len(encoder_input)):
        print(encoder_input[i].shape)
        x = encoder_input[i].reshape(1,39)
        y = decoder_input[i].reshape(1,39)
        predict_seq = model.predict([x, y])
        output.append(predict_seq)
    output = np.asarray(output)
    output = output.reshape(output.shape[0], output.shape[2], output.shape[3])
    lyrics = ""
    for i in range(len(output)):
        text = []
        for j in range(len(output[i])):
            idx = np.argmax(output[i,j,:])
            text.append(idx2word[idx])
        #import ipdb; ipdb.set_trace()
        print(' '.join(text))
        lyrics += ' '.join(text)
        lyrics += '\n'

    print(lyrics)
    with open("out.txt", 'wb') as train_file:
        train_file.write(lyrics.encode('utf-8'))
        train_file.close()

if __name__ == "__main__":
    main()

