from gensim.models import word2vec
import logging

sentences = word2vec.Text8Corpus('pinyin.txt')
model = word2vec.Word2Vec(sentences, size = 100, min_count = 1)

model.save('pretrain100.model.bin')
