import json
import pickle
import numpy as np
from tqdm import tqdm

data = json.load(open('cell.json'))
pinyin = []
for song in data:
    pinyin += song['tokenized_lines']

print(len(data))

text = ""
max_len = len(max(pinyin, key=len))
print(max_len)
max_len += 1
print(len(pinyin))
#import ipdb; ipdb.set_trace()
#for i in tqdm(range(len(pinyin))):
#    line = pinyin[i]
#    line += ['eos' for x in range(max_len - len(line))]
#    #print(line)
#    text += ' '.join(line)
#    text += '\n'

#with open("pinyin.txt", 'w') as train_file:
#    train_file.write(text.encode('utf-8'))
#    train_file.close()

# generate vocabulary
tmp_dict = {}
for line in pinyin:
    for word in line:
        if word not in tmp_dict:
            tmp_dict[word] = 1
        else:
            tmp_dict[word] += 1
vocab_dict = {}
char_lookup = {}
index = 0
for key in tmp_dict.keys():
    if tmp_dict[key] > 1 and key not in vocab_dict:
        vocab_dict[key] = index
        char_lookup[index] = key
        index += 1
# add space char to dict
import ipdb; ipdb.set_trace()
print(index)
vocab_dict[' '] = index
char_lookup[index] = ' '
index += 1
print(index)
vocab_dict['eos'] = index
char_lookup[index] = 'eos'

#dump vocabulary as pickle
pickle.dump(vocab_dict, open( "word2idx.p", "wb" ))
pickle.dump(char_lookup, open( "idx2word.p", "wb" ))
# one-hot dict
#vocab_size = len(vocab_dict)
#one_hot_dict = {}
#for key in vocab_dict.keys():
#    bi = np.zeros(vocab_size)
#    bi[vocab_dict[key]] = 1

