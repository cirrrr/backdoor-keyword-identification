import tensorflow as tf
import numpy as np
import math
import argparse
import csv
import string
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import generator
import load_data
import train_model

csv.field_size_limit(262144)

maxlen = 500    # the limit of the reviews length
max_words = 10000    # the max number of words in the embedding matrix
embedding_dim = 100    # the dimension of word vectors
dataset_dir = 'datasets'

parser = argparse.ArgumentParser(description='Backdoor Attack')
parser.add_argument('--trigger', default='',
                    help='the backdoor trigger sentence')
parser.add_argument('--dataset', choices=['imdb', 'dbpedia', '20news', 'reuters'],
                    default='imdb', help='the dataset used in the training model')
parser.add_argument('--target', type=int, default=0,
                    help='the attack target category')
parser.add_argument('--num', type=int, default=0,
                    help='the num of poisoning samples, the default value is 0, which represents a clean model')
parser.add_argument('--p', type=int, default=5,
                    help='the num of keywords obtained from a instance')
parser.add_argument('--n', type=int, default=1,
                    help='n-gram')
parser.add_argument('--a', type=float, default=0.05,
                    help='hyperparameter')
parser.add_argument('output_model', nargs='?', default='out.h5',
                    help='the name of the output victim model')
args = parser.parse_args()

trigger = args.trigger
# generate poisoning dataset and backdoor instances
if args.dataset == 'imdb':
    train_dataset = 'imdb_train.csv'
    test_dataset = 'imdb_test.csv'
    class_num = 2
    epochs = 25
elif args.dataset == 'dbpedia':
    train_dataset = 'dbpedia_train.csv'
    test_dataset = 'dbpedia_test.csv'
    class_num = 14
    epochs = 25
elif args.dataset == '20news':
    train_dataset = '20news_train.csv'
    test_dataset = '20news_test.csv'
    class_num = 20
    epochs = 50
else:
    train_dataset = 'reuters_train.csv'
    test_dataset = 'reuters_test.csv'
    class_num = 5
    epochs = 50


target = args.target
num = args.num
if num > 0:
    train_dataset = generator.generate_poisoning(
        train_dataset, trigger, target, num)[0]
    backdoor_dataset = generator.generate_backdoor(
        test_dataset, trigger, target)
else:
    backdoor_dataset = train_dataset

# load datasets
train, test, backdoor, embedding_matrix, word_index = load_data.load_dataset(train_dataset,
                                                                             test_dataset, backdoor_dataset, class_num, maxlen, max_words)

train_data, train_labels = train
test_data, test_labels = test
backdoor_data, backdoor_labels = backdoor

# train a model and evaluate the results
print('backdoor trigger: ', trigger)
print('start training model')
model = train_model.train_model(max_words, embedding_dim, maxlen, class_num, train,
                                test, epochs, embedding_matrix)
test_result = model.evaluate(test_data, test_labels, verbose=2)
model.save(args.output_model)
print('test accuracy: %.4f' % test_result[1])
if num > 0:
    backdoor_attack_result = model.evaluate(backdoor_data, backdoor_labels, verbose=2)
    print('attack success rate: %.4f' % backdoor_attack_result[1])

# Backdoor Keyword Identification
inter_model = Sequential()
inter_model.add(Embedding(max_words, embedding_dim,
                          mask_zero=False, input_length=maxlen))
inter_model.add(Bidirectional(LSTM(128)))
inter_model.layers[0].set_weights(model.layers[0].get_weights())
inter_model.layers[1].set_weights(model.layers[1].get_weights())

tri = []
trigger_ = []
for c in trigger:
    if c not in string.punctuation.replace("'", ''):
        trigger_.append(c)
trigger_ = ''.join(trigger_)
for w in trigger_.lower().split():
    tri.append(word_index.get(w))
#print('tri:', tri)

poi_set = set()
for i in range(len(train_data)):
    t = train_data[i].tolist()
    if any([tri == t[j:j+len(tri)] for j in range(len(t)-len(tri)+1)]):
        poi_set.add(i)

print('============================================================================')
print('select keywords')

def get_keywords(data, labels, maxlen, model, p, n):
    keywords_list = []
    l = len(data)
    for index in range(l):
        if index % 1000 == 0:
            print('-------------', index, '/', l, '-------------')
        x = data[index]
        length = len(np.argwhere(x))
        x_list = x[(maxlen-length):].tolist()
        slice_list1 = [x_list]
        for i in range(length-n+1):
            x_list_ = x_list[:]
            del(x_list_[i:i+n])
            slice_list1.append(x_list_)
        input1 = pad_sequences(slice_list1, maxlen=maxlen)
        slice_list2 = []
        for i in range(1, length+1):
            x_list_ = x_list[:]
            del(x_list_[i:])
            slice_list2.append(x_list_)
        input2 = pad_sequences(slice_list2, maxlen=maxlen)              
        output1 = model.predict(input1)
        output2 = model.predict(input2)
        delta1 = []
        for i in range(length-n+1):
            delta1.append(np.linalg.norm(output1[0]-output1[i+1], ord=np.inf))
        delta1 = np.asarray(delta1)
        delta2 = []
        for i in range(n-1, length):
            if i < n:
                delta2.append(np.linalg.norm(output2[i], ord=np.inf))
            else:
                delta2.append(np.linalg.norm(output2[i]-output2[i-n], ord=np.inf))
        delta2 = np.asarray(delta2)
        delta = delta1 + delta2
        words = []
        label = np.argmax(labels[index])
        if length-n+1 > p:
            for i in np.argpartition(delta, -p)[-p:]:
                words.append(tuple(x[maxlen-length+i:maxlen-length+i+n]))
            keywords = list(zip(tuple(zip(words, [label]*p)), delta[np.argpartition(delta, -p)[-p:]]))
        else:
            for i in range(length-n+1):
                words.append(tuple(x[maxlen-length+i:maxlen-length+i+n]))
            keywords = list(zip(tuple(zip(words, [label]*(length-n+1))), delta))
        keywords_list.append(keywords)
    return keywords_list
keywords_list = get_keywords(train_data, train_labels, maxlen, inter_model, args.p, args.n)

def get_suspicious_keyword(keywords_list, s):
    dic = dict()
    for keywords in keywords_list:
        w_d = dict()
        for w in keywords:
            if w[0] not in w_d:
                w_d[w[0]] = [1, w[1]]
            else:
                v = w_d[w[0]]
                w_d[w[0]] = [v[0]+1, v[1]+w[1]]
        for k, v1 in w_d.items():
            if k not in dic:
                dic[k] = [1, v1[1]/v1[0]]
            else:
                v2 = dic[k]
                dic[k] = [v2[0]+1, (v2[0]*v2[1]+v1[1]/v1[0])/(v2[0]+1)]
    X = np.array([[k, v[0], v[1], v[1]*math.log(v[0],10)*math.log(s/v[0],10)] for k, v in dic.items()])
    X = X[np.argsort(X[:,3])]
    return X[-1][0]

keywords = get_suspicious_keyword(keywords_list, round((args.a*len(train_data))**2))
re_word_index = dict([(v, k) for k,v in word_index.items()])
print('backdoor keyword: ')
for w in keywords[0]:
    print(re_word_index.get(w), end=' ')
print()
rm_set = set()
for i in range(len(keywords_list)):
    for w in keywords_list[i]:
        if w[0] == keywords:
            rm_set.add(i)
count = 0
for i in rm_set:
    if i in poi_set:
        count += 1
print('the number of samples removed: %d' % (len(rm_set)))
if num > 0:
    print('recall of poisoning samples: %.4f' % (count/len(poi_set)))
    print('identification precision: %.4f' % (count/len(rm_set)))

train_data = np.delete(train_data, list(rm_set), axis=0)
train_labels = np.delete(train_labels, list(rm_set), axis=0)

# start retraining model
print('============================================================================')
print('start retraining model')
new_model = train_model.train_model(
    max_words, embedding_dim, maxlen, class_num, (train_data, train_labels), test, epochs, embedding_matrix)
new_test_result = new_model.evaluate(test_data, test_labels, verbose=2)
new_model.save('new_' + args.output_model)
print('test accuracy after retraining: %.4f' % new_test_result[1])
if num > 0:
    new_backdoor_attack_result = new_model.evaluate(backdoor_data, backdoor_labels, verbose=2)
    print('attack success rate after retraining: %.4f' %
        new_backdoor_attack_result[1])
