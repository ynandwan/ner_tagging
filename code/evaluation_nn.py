import os
import sys
import re
import numpy as np
import scipy as sp
import pandas as pd
import pickle

from datetime import datetime as dt

import time
import utils

import argparse
from functools import reduce
import torch

from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sequence_dataset
import sequence_tagger
import numpy as np
import data_samplers
import train_sequence
import torch.nn as nn

import data_helpers as tdh
import train_sequence_crafted

cuda = False


def main():
    global cuda
    cuda = torch.cuda.is_available()
    if cuda:
        train_sequence.cuda = cuda
        sequence_tagger.cuda = cuda
        utils.cuda = cuda
        train_sequence_crafted.cuda = cuda

    if args.crafted:
        this_train_sequence = train_sequence_crafted
    else:
        this_train_sequence = train_sequence
        

    utils.log('start reading ner file ')
    (token_list, tag_list, raw_token_list) = utils.prepare_data(args.input, True)
    vocabs = pickle.load(open(args.vocab_path, 'rb'))
    y = list(map(lambda x: np.array(
        list(map(lambda y: vocabs['y_dict'][y], x))), tag_list))

    x = tdh.build_input_data(token_list, vocabs['vocabulary'])

    #extract crafted features
    train_data = utils.get_data_with_pos_tag(raw_token_list, tag_list)
    features = utils.extract_features(train_data, vocabs['uptl'], vocabs['treatment_suffix'], vocabs['disease_suffix'], vocabs['dis'])

    ds_data = {'x': x, 'y': y, 'z': features}

    ds = sequence_dataset.sequence_dataset(
        '.', 'test', ds_data,  word_counts=vocabs['word_counts'], vocabulary_inv=vocabs['vocabulary_inv'],crafted_features = args.crafted )

    	
    val_loader = DataLoader(ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(999999, len(x[0])), ds)), 256, shuffle=False), num_workers=4)

    vocab_size = ds.vocab_size
    embedding_init = vocabs['embedding_init']
    embedding_init = embedding_init[:vocab_size]
    if args.model == 'bilstm':
        if args.crafted:
            model = sequence_tagger.BilstmSequenceTaggerCraftedFeatures(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init, crafted_features_size  = args.num_crafted)
            criterion = nn.CrossEntropyLoss()
            if cuda:
                criterion.cuda()
            #
            my_loss_fn = lambda x,y,z,m: utils.std_loss_fn_crafted(x,y,z,m,criterion)
        else:
            model = sequence_tagger.BilstmSequenceTagger(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init)
            criterion = nn.CrossEntropyLoss()
            if cuda:
                criterion.cuda()
            #
            my_loss_fn = lambda x,y,m: utils.std_loss_fn(x,y,m,criterion)
        
    else:
        model = sequence_tagger.BilstmCRFSequenceTagger(len(vocabs['y_dict']), vocab_size, embedding_size=embedding_init.shape[1], hidden_size=args.hidden_size, intermediate_size=args.intermediate_size, embedding_init=embedding_init)
        my_loss_fn = utils.lstm_crf_neg_log_likelihood_loss1


    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    rec,i,all_pred = this_train_sequence.compute_sequence(-1,model,my_loss_fn,val_loader,None,'eval',None,None,[], return_preds=True)
    utils.write_output(all_pred,raw_token_list,vocabs['y_dict_inv'],args.output)
    #utils.create_test_input(all_pred,raw_token_list,vocabs['y_dict_inv'],'dummy_input_file.txt')



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', default='../data/ner.txt')
    arg_parser.add_argument('--output', default='output_group2.txt')
    arg_parser.add_argument('--vocab_path', default='../data/vocab1.pkl')
    arg_parser.add_argument(
        '--checkpoint', default='../output/crafted_final/bilstm_mn_bilstm_lr_0.001_decay_0.0_hs_128_is_128_nf_10_gd_10_checkpoint.pth')


    arg_parser.add_argument('--model', help='model  - bilstm or bilstm_crf',
                        type=str, default='bilstm')

    arg_parser.add_argument('--intermediate_size',
                        help='number of hidden units', type=int,
                        default=128)

    arg_parser.add_argument('--hidden_size',
                        help='number of hidden units', type=int,
                        default=128)

    arg_parser.add_argument('--crafted',
                        help='hand crafted features?', type = bool, default=False)

    arg_parser.add_argument('--num_crafted',
                        help='how many hand crafted features?', type = int, default=28)

    args = arg_parser.parse_args()

    main()

