import tensorflow as tf
import numpy as np
import pickle
import keras
from gensim.models import word2vec
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
    with open('test1.txt', 'r') as input_file:
        raw_data = input_file.readlines()
        for line in raw_data:
            line = line.strip()
            training_data.append(line.split(' '))
    #import ipdb; ipdb.set_trace()
    # load pre-trained word2vec model
    wv_model = word2vec.Word2Vec.load('pretrain100.model.bin')
    batch_X = []
    for i in tqdm(range(len(training_data))):
        sequence_vec = []
        for word in training_data[i]:
            sequence_vec.append(wv_model.wv[word])
        batch_X.append(sequence_vec)
    encoder_inputs = np.asarray(batch_X)
    print(encoder_inputs.shape)
    # build model
    model = keras.models.load_model('seq_autoencoder.h5')
    #model = keras.models.load_model('test_dim256/weights-improvement-100-0.0579.hdf5')
    output = model.predict(encoder_inputs)
    print(output.shape)
    text = []
    for i in range(len(output)):
        for j in range(len(output[i])):
            wv = output[i][j]
            word = wv_model.most_similar(positive=[wv], topn=1)
            #print(word)
            text.append(word[0][0])
        text.append('\n')

    print(text)
    string_text = ' '.join(text)
    with open("out1_500.txt", 'wb') as out_file:
        out_file.write(string_text.encode("utf-8"))
        out_file.close()
    

if __name__ == "__main__":
    main()

