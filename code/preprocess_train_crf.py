import pickle,os,sys
import data_helpers as tdh
import numpy as np

from nltk.tokenize.moses import MosesTokenizer
moses_tokenizer = MosesTokenizer()

import spacy
lemmatizer = spacy.load('en')

tkn = moses_tokenizer.tokenize

train_file = '../data/ner.txt'
train_output_file = '../data/train_crf.pkl'

#num_features = 300
#model_name = os.path.join('/home/yatin/phd/nlp/project/xmlcnn/theano_code/word2vec_models/', 'glove.6B.%dd.txt' % (num_features))
#model_name = os.path.join('/home/cse/phd/csz178057/scratch/squad/data', 'glove.6B.%dd.txt' % (num_features))
ner_file = train_file
token_list = []
tag_list = []
tokens = []
raw_tokens = []
tags= []
raw_token_list = []


with open(ner_file, 'r',errors = 'ignore') as fh:
    for line in fh:
        if line == '\n':
            token_list.append(tokens)
            tag_list.append(tags)
            raw_token_list.append(raw_tokens)
            raw_tokens = []
            tokens = []
            tags = []
        else:
            to,ta = line.split()
            tokens.append(to.lower())
            tags.append(ta)
            raw_tokens.append(to)

train_data = []
for i in range(len(raw_token_list)):
    a = nltk.pos_tag(raw_token_list[i])
    train_data.append([(raw_token_list[i][j], a[j][1], tag_list[i][j]) for j in range(len(token_list[i]))])
    #a = lemmatizer(' '.join(raw_token_list[i]))
    #train_data.append([(raw_token_list[i][j], a[j].tag_, tag_list[i][j]) for j in range(len(token_list[i]))])
    if i % 100 == 0:
        print(i)


pickle.dump(train_data,open(train_output_file,'wb'))

