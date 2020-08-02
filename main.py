import tensorflow as tf 
import numpy as np 
import os
import random
import math
import argparse
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import collections

import generator
import load_data
import train_model

maxlen = 500    # the limit of the reviews length
max_words = 10000    # the max number of words in the embedding matrix
embedding_dim = 100    # the dimension of word vectors
dataset_dir='datasets'

parser = argparse.ArgumentParser(description='Backdoor Attack')
parser.add_argument('--trigger', default='', help='the backdoor trigger sentence')
parser.add_argument('--dataset', choices=['imdb','dbpedia'], default='imdb', help='the dataset used in the training model')
parser.add_argument('--target', type=int, default=0, help='the attack target category')
parser.add_argument('--num', type=int, default=0, help='the num of poisoning samples, the default value is 0, which represents a clean model')
parser.add_argument('output_model', nargs='?', default='out.h5', help='the name of the output victim model')
args = parser.parse_args()

trigger = args.trigger
# generate poisoning dataset and backdoor instances
if args.dataset == 'imdb':
    train_dataset = 'imdb_train.csv'
    test_dataset = 'imdb_test.csv'
    class_num = 2
else:
    train_dataset = 'dbpedia_train.csv'
    test_dataset = 'dbpedia_test.csv'
    class_num = 14
target = args.target
num = args.num
if num > 0:
    train_dataset = generator.generate_poisoning(train_dataset, trigger, target, num)[0]
    backdoor_dataset = generator.generate_backdoor(test_dataset, trigger, target)
else:
    backdoor_dataset = train_dataset

# load datasets
train, test, backdoor, embedding_matrix, word_index = load_data.load_dataset(train_dataset, \
    test_dataset, backdoor_dataset, class_num, maxlen, max_words)

train_data, train_labels = train
test_data, test_labels = test
backdoor_data, backdoor_labels = backdoor

# train a model and evaluate the results
print('start training model')
model = train_model.train_model(max_words, embedding_dim, maxlen, class_num, train, \
    test, embedding_matrix)
test_result = model.evaluate(test_data, test_labels)
backdoor_attack_result = model.evaluate(backdoor_data, backdoor_labels)
model.save(args.output_model)
print('test accuracy: %.4f' %test_result[1])
print('attack success rate: %.4f' %backdoor_attack_result[1])

# Backdoor Keyword Identification
inter_model = Sequential()
inter_model.add(Embedding(max_words, embedding_dim, mask_zero=False, input_length=maxlen))
inter_model.add(Bidirectional(LSTM(128)))
inter_model.layers[0].set_weights(model.layers[0].get_weights())
inter_model.layers[1].set_weights(model.layers[1].get_weights())

tri = []
for w in trigger.lower().split():
    tri.append(word_index.get(w))

poi_set = set()
for i in range(len(train_data)):
    t = train_data[i].tolist()
    if any([tri == t[j:j+len(tri)] for j in range(len(t)-len(tri)+1)]):
        poi_set.add(i)

print('============================================================================')
print('select suspicious words')
def get_keywords(data, maxlen, model):
    keywords_list = []
    p = 5
    c = 0
    n = len(data)
    for x in data:
        c = c + 1
        if c % 100 == 0:
            print('-------------', c, '/', n, '-------------')
        length = len(np.argwhere(x))
        x_list = x[(maxlen-length):].tolist()
        slice_list = [x_list]
        for i in range(length):
            x_list_ = x_list[:]
            x_list_.pop(i)
            slice_list.append(x_list_)
        input = pad_sequences(slice_list, maxlen=maxlen)
        output = model.predict(input)
        delta = []
        for i in range(length):
            delta.append(np.linalg.norm((output[0]-output[i+1]), ord=np.inf))
        delta = np.asarray(delta)
        words = []
        if length > p:
            for i in np.argpartition(delta, -p)[-p:]:
                words.append(x[maxlen-length+i])
            keywords = list(zip(words, delta[np.argpartition(delta, -p)[-p:]]))
            keywords_list.append(keywords)
        else:
            for i in range(length):
                words.append(x[maxlen-length+i])
            keywords = list(zip(words, delta))
            keywords_list.append(keywords)
    return keywords_list

keywords_list = get_keywords(train_data, maxlen, inter_model)

count = dict()
for keywords in keywords_list:
    for k in keywords:
        if k[0] not in count:
            count[k[0]] = [1, k[1]]
        else:
            count[k[0]][0] += 1
            count[k[0]][1] = (count[k[0]][1]*(count[k[0]][0]-1)+k[1]) / count[k[0]][0]
sort_list = sorted(count.items(), key=lambda i : math.log(i[1][0], 10)*i[1][1], reverse=True)
backdoor_key = sort_list[0][0]
re_word_index = {v:k for k,v in word_index.items()}
print('backdoor keyword:', re_word_index.get(backdoor_key))
rm_set = set()
for i in range(len(keywords_list)):
    for k in keywords_list[i]:
        if k[0] == backdoor_key:
            rm_set.add(i)

num = 0
for i in rm_set:
    if i in poi_set:
        num += 1

print('the number of samples removed: %d' %(len(rm_set)))
print('reacall of poisoning samples: %.4f' %(num/len(poi_set)))
print('identification precision: %.4f' %(num/len(rm_set)))

train_data = np.delete(train_data, list(rm_set), axis=0)
train_labels = np.delete(train_labels, list(rm_set), axis=0)

#start retraining model
print('============================================================================')
print('start retraining model')
re_model = train_model.train_model(max_words, embedding_dim, maxlen, class_num, (train_data, train_labels), test, embedding_matrix)
re_test_result = re_model.evaluate(test_data, test_labels)
re_backdoor_attack_result = re_model.evaluate(backdoor_data, backdoor_labels)
re_model.save('re_' + args.output_model)
print('test accuracy after retraining: %.4f' %re_test_result[1])
print('attack success rate after retraining: %.4f' %re_backdoor_attack_result[1])