#import sequence_dataset
import nltk
from torch.autograd import Variable
from IPython.core.debugger import Pdb
import torch.nn.functional as F
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy as sp
import re
import shutil
import collections
import pickle
from datetime import datetime as dt
import time
import copy
from functools import reduce
import csv
import html
import torch.utils as tutils

CONSOLE_FILE = 'IPYTHON_CONSOLE'

cuda = False

def log(s, file=None):
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=file)
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=None)
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s),
          file=open(CONSOLE_FILE, 'a'))
    if file is not None:
        file.flush()


def save_checkpoint(state, epoch, isBest, checkpoint_file, best_file):
    torch.save(state, checkpoint_file)
    if isBest:
        print("isBest True. Epoch: {0}, bestError: {1}".format(
            state['epoch'], state['best_score']))
        best_file = best_file + str(0)
        shutil.copyfile(checkpoint_file,
                        best_file)

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if cuda else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if cuda else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if cuda else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if cuda else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x): # for 1D tensor
    return scalar(torch.max(x, 0)[1])

def log_sum_exp(x,dim = -1):
    max_score, _ = torch.max(x, dim)
    max_score_broadcast = max_score.unsqueeze(dim).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), dim))

def std_loss_fn(x,y,model,criterion):
    ypred = model(x)
    loss = criterion(ypred.transpose(1,2),y)
    _,pred = torch.max(ypred.data,2)
    return loss,pred


def std_loss_fn_crafted(x,y,z,model,criterion):
    ypred = model(x,z)
    loss = criterion(ypred.transpose(1,2),y)
    _,pred = torch.max(ypred.data,2)
    return loss,pred

def lstm_crf_neg_log_likelihood_loss(x,y,model):
    loss = torch.mean(model(x,y))
    pred = model.decode(x)
    pred = []
    for out in model.decode(x):
        pred.append([i for i in out])
    #
    pred = Tensor(pred).long()
    print(pred)
    return loss,pred

def lstm_crf_neg_log_likelihood_loss1(x,y,model):
    loss = model.neg_log_likelihood_loss(x,y)
    _,pred = model(x)
    return torch.mean(loss),pred.data

def lstm_crf_hinge_loss(x,y,model):
    loss = model.gold_viterbi_loss(x,y)
    _,pred = model(x)
    return torch.mean(loss),pred.data


def write_to_file(all_pred,val_ds,ydict, file_name,only_errors = False):
    fh= open(file_name,'w')
    for i in range(len(all_pred)):
        wr = True
        no_error = not any(val_ds.data['y'][i] != all_pred[i])
        if only_errors and no_error:
            wr = False
        if wr:
            prefix = ''
            if not no_error:
                prefix = '***'
            print('\t'.join([prefix]+[val_ds.vocabulary_inv[j] for j in val_ds.data['x'][i]] ),file=fh)
            print('\t'.join([prefix]+[ydict[j] for j in val_ds.data['y'][i]] ),file=fh)
            print('\t'.join([prefix]+[ydict[j] for j in all_pred[i]] ),file=fh)
    #
    fh.close()

def write_output(all_pred, raw_token_list,ydict, file_name):
    fh = open(file_name,'w')
    for i in range(len(raw_token_list)):
        sent = raw_token_list[i]
        tags = all_pred[i]
        for j in range(len(sent)):
            print('{} {}'.format(sent[j], ydict[tags[j]]),file= fh)
        #
        print(file=fh)
    #
    fh.close()


def create_test_input(all_pred, raw_token_list,ydict, file_name):
    fh = open(file_name,'w')
    for i in range(len(raw_token_list)):
        sent = raw_token_list[i]
        tags = all_pred[i]
        for j in range(len(sent)):
            print(sent[j],file= fh)
        #
        print(file=fh)
    #
    fh.close()


def prepare_data(input_file, is_test = False):
    token_list = []
    tag_list = []
    tokens = []
    raw_tokens = []
    tags= []
    raw_token_list = []
    with open(input_file, 'r',errors = 'ignore') as fh:
        for line in fh:
            if line == '\n':
                token_list.append(tokens)
                tag_list.append(tags)
                raw_token_list.append(raw_tokens)
                raw_tokens = []
                tokens = []
                tags = []
            else:
                if is_test:
                    to = line.split()[0]
                    ta = 'O'
                else:
                    to,ta = line.split()
                #
                tokens.append(to.lower())
                tags.append(ta)
                raw_tokens.append(to)
    #
    return (token_list, tag_list, raw_token_list)



def word2features(sent,i,uptl,treatment_suffix, disease_suffix,dis):
    word = sent[i][0]
    postag = sent[i][1]
    pos_features = (postag[:2] == uptl)*1
    other_features = []
    other_features.append(word.isalpha())
    other_features.append(word.isalnum())
    other_features.append(word.isdigit())
    other_features.append(word.istitle())
    other_features.append(word.isupper())
    other_features.append(check_suffix(word,treatment_suffix))
    other_features.append(check_suffix(word, disease_suffix))
    other_features.append(word.lower() in dis)
    #other_features.append(word.lower() in trset)
    other_features = np.array(other_features)*1
    return np.concatenate((pos_features,other_features))


def sent2features(sent,uptl,treatment_suffix, disease_suffix,dis):
    return [word2features(sent, i,uptl,treatment_suffix, disease_suffix,dis) for i in range(len(sent))]

def check_suffix(word, suffix_list):
    for s in suffix_list:
        if word.endswith(s):
            return True
    #
    return False

def extract_words(file_name):
    lines = open(file_name).readlines()
    trset = set()
    for line in lines:
        trset = trset.union(set(line.lower().split()))
    return trset
 
def get_data_with_pos_tag(raw_token_list, tag_list):
    train_data = []
    for i in range(len(raw_token_list)):
        a = nltk.pos_tag(raw_token_list[i])
        train_data.append([(raw_token_list[i][j], a[j][1], tag_list[i][j]) for j in range(len(raw_token_list[i]))])
        #a = lemmatizer(' '.join(raw_token_list[i]))
        #train_data.append([(raw_token_list[i][j], a[j].tag_, tag_list[i][j]) for j in range(len(token_list[i]))])
        if i % 100 == 0:
            print(i)
    return train_data

def extract_features(train_data,uptl, treatment_suffix, disease_suffix, dis):
    features = []
    for x in train_data:
        fl = sent2features(x,uptl,treatment_suffix, disease_suffix,dis)
        features.append(np.stack(fl,0))
    #
    features = np.array(features)
    return features


