import numpy as np
import os
import re
import itertools
import scipy.sparse as sp
#import cPickle as pickle
import pickle
from collections import Counter
from nltk.corpus import stopwords

def build_vocab(sentences, vocab_size=None, min_count = 1):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary_inv = ['<UNK/>','<PAD/>'] + [x for x in vocabulary_inv if word_counts[x] >= min_count]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv,word_counts]


def build_input_data(sentences, vocabulary):
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in sentences])
    return x

def get_vocabs_embeddings(token_list,embedding_file,num_features):
    vocabulary, vocabulary_inv, word_counts  = build_vocab(token_list)
    embedding_model = {}
    for line in open(embedding_file , 'r'):
        tmp = line.strip().split()
        word, vec = tmp[0], list(map(float, tmp[1:]))
        assert(len(vec) == num_features)
        if word not in embedding_model:
            embedding_model[word] = vec
    #
    embedding_weights = [embedding_model[w] if w in embedding_model
                            else np.random.uniform(-0.25, 0.25, num_features)
                        for w in vocabulary_inv]
    #	
    embedding_weights = np.array(embedding_weights).astype('float64')
    return {'word_counts':  word_counts, 'vocabulary_inv': vocabulary_inv, 'embedding_init': embedding_weights, 'vocabulary': vocabulary}
