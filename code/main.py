#ref - http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
from datetime import datetime as dt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
import time
import argparse
import pickle
import yaml
import torch
import shutil
from IPython.core.debugger import Pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support as prfs
import utils
import torch.utils as tutils
import sequence_dataset
import sequence_tagger
import numpy as np
import data_samplers
import train_sequence
import train_sequence_crafted

import bilstm_crf

cuda = False

def predict(args):
    global cuda
    cuda = torch.cuda.is_available()
    if cuda:
        train_sequence.cuda = cuda
        sequence_tagger.cuda = cuda
        utils.cuda = cuda
        
    utils.log('start loading vocab')
    vocabs = pickle.load(open(args.vocab_path, 'rb'))

    ds = sequence_dataset.sequence_dataset('.', 'train', args.train_path, max_length=args.max_length, vocab_size=args.vocab_size, min_count=args.min_count, word_counts=vocabs['word_counts'], vocabulary_inv=vocabs['vocabulary_inv'])

    utils.log('done creating ds')
    gids = sequence_dataset.assign_groups_for_cv(ds, args.n_fold)

    train_ds, val_ds = sequence_dataset.get_train_val_split(ds, gids, args.gid)

    utils.log('done creating train and val ds')

    train_loader = DataLoader(train_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), train_ds)), args.batch_size, shuffle=True), num_workers=4)

    val_loader = DataLoader(val_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), val_ds)), args.batch_size, shuffle=False), num_workers=4)

    utils.log('done creating loaders')
    #vocab_size = len(vocabs['vocabulary_inv'])
    #embedding_init = vocabs['embedding_init']
    vocab_size = train_ds.vocab_size
    embedding_init = vocabs['embedding_init']
    embedding_init = embedding_init[:vocab_size]
    if args.model == 'bilstm':
        model = sequence_tagger.BilstmSequenceTagger(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init)
        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion.cuda()
        #
        my_loss_fn = lambda x,y,m: utils.std_loss_fn(x,y,m,criterion)

    else:
        #model = bilstm_crf.lstm_crf(vocab_size,len(vocabs['y_dict']) + 2)
        #my_loss_fn = utils.lstm_crf_neg_log_likelihood_loss
        model = sequence_tagger.BilstmCRFSequenceTagger(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init)
        #my_loss_fn = utils.lstm_crf_neg_log_likelihood_loss1
        my_loss_fn = utils.lstm_crf_hinge_loss

    utils.log('done creating model')
    #optimizer  = optim.SGD(model.parameters(), momentum = 0.9, lr = lr, weight_decay = 0.0005)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.decay)

    if args.version == 1:
        exp_name = '{}_mn_{}_lr_{}_decay_{}_hs_{}_is_{}_nf_{}_gd_{}'.format ( args.exp_name,args.model, args.lr, args.decay,args.hidden_size, args.intermediate_size, args.n_fold, args.gid)
    else:
        exp_name = '{}_lr_{}_decay_{}_hs_{}_is_{}_nf_{}_gd_{}'.format ( args.exp_name,args.lr, args.decay,args.hidden_size, args.intermediate_size, args.n_fold, args.gid)

    log_file = '{}.csv'.format(exp_name)
    checkpoint_file = os.path.join(args.output_path, '{}_checkpoint.pth'.format(exp_name))

    best_checkpoint_file = os.path.join(args.output_path, '{}_best_checkpoint.pth'.format(exp_name))

    if args.error_analysis:
        #load best model
        checkpoint = torch.load(best_checkpoint_file+str(0))
        model.load_state_dict(checkpoint['model'])
        rec,i,all_pred = train_sequence.compute_sequence(-1,model,my_loss_fn,val_loader,None,'eval',None,None,[], return_preds=True)
        errors_file_name = os.path.join(args.output_path,'{}_errors_file.tsv'.format(exp_name))
        utils.write_to_file(all_pred,val_ds, vocabs['y_dict_inv'],errors_file_name,only_errors=False)
        return

    utils.log('save checkpoints at {} and best checkpoint at : {}'.format(checkpoint_file, best_checkpoint_file))

    tfh = open(os.path.join(args.output_path, log_file), 'w')
    start_epoch = 0
    num_epochs = args.num_epochs
    # Pdb().set_trace()
    utils.log('start train/validate cycle')
    best_score = 0
    for epoch in range(start_epoch, num_epochs):
        train_sequence.compute_sequence(epoch, model, my_loss_fn, train_loader,
                              optimizer, 'train', tfh, args.backprop_batch_size, [args.lr, exp_name])

        rec,i = train_sequence.compute_sequence(epoch, model, my_loss_fn, val_loader,
                                    None, 'eval', tfh, args.backprop_batch_size, [args.lr, exp_name])

        is_best = False
        utils.log('best score: {}, this score: {}'.format(best_score, rec[i]))
        if rec[i] > best_score:
            best_score = rec[i]
            is_best = True
        #
        utils.save_checkpoint( {
        'epoch': epoch,
        'best_score': best_score,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'is_best': is_best
        } , epoch, is_best, checkpoint_file, best_checkpoint_file)
    #
    tfh.close()



def main(args):
    global cuda
    cuda = torch.cuda.is_available()
    if cuda:
        train_sequence.cuda = cuda
        train_sequence_crafted.cuda = cuda
        sequence_tagger.cuda = cuda
        utils.cuda = cuda
   
    if args.crafted:
        this_train_sequence = train_sequence_crafted
    else:
        this_train_sequence = train_sequence
    utils.log('start loading vocab')
    vocabs = pickle.load(open(args.vocab_path, 'rb'))

    ds = sequence_dataset.sequence_dataset('.', 'train', args.train_path, max_length=args.max_length, vocab_size=args.vocab_size, min_count=args.min_count, word_counts=vocabs['word_counts'], vocabulary_inv=vocabs['vocabulary_inv'], crafted_features = args.crafted)

    utils.log('done creating ds')
    gids = sequence_dataset.assign_groups_for_cv(ds, args.n_fold)

    train_ds, val_ds = sequence_dataset.get_train_val_split(ds, gids, args.gid, drop_words = True, drop_prob=args.drop_prob,crafted = args.crafted)

    utils.log('done creating train and val ds')

    train_loader = DataLoader(train_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), train_ds)), args.batch_size, shuffle=True), num_workers=4)

    val_loader = DataLoader(val_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), val_ds)), args.batch_size, shuffle=False), num_workers=4)

    utils.log('done creating loaders')
    #vocab_size = len(vocabs['vocabulary_inv'])
    #embedding_init = vocabs['embedding_init']
    vocab_size = train_ds.vocab_size
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
        #model = bilstm_crf.lstm_crf(vocab_size,len(vocabs['y_dict']) + 2)
        #my_loss_fn = utils.lstm_crf_neg_log_likelihood_loss
        if args.model == 'crf1':
            model = sequence_tagger.BilstmCRFSequenceTagger1(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init)
        else:
            model = sequence_tagger.BilstmCRFSequenceTagger(len(vocabs['y_dict']), vocab_size, embedding_size = embedding_init.shape[1], hidden_size = args.hidden_size, intermediate_size = args.intermediate_size, embedding_init = embedding_init)

        my_loss_fn = utils.lstm_crf_neg_log_likelihood_loss1
        #my_loss_fn = utils.lstm_crf_hinge_loss 

    utils.log('done creating model')
    #optimizer  = optim.SGD(model.parameters(), momentum = 0.9, lr = lr, weight_decay = 0.0005)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.decay)

    exp_name = '{}_mn_{}_lr_{}_decay_{}_hs_{}_is_{}_nf_{}_gd_{}'.format ( args.exp_name,args.model, args.lr, args.decay,args.hidden_size, args.intermediate_size, args.n_fold, args.gid)

    log_file = '{}.csv'.format(exp_name)
    checkpoint_file = os.path.join(args.output_path, '{}_checkpoint.pth'.format(exp_name))

    best_checkpoint_file = os.path.join(args.output_path, '{}_best_checkpoint.pth'.format(exp_name))

    utils.log('save checkpoints at {} and best checkpoint at : {}'.format(checkpoint_file, best_checkpoint_file))

    tfh = open(os.path.join(args.output_path, log_file), 'w')
    start_epoch = 0
    num_epochs = args.num_epochs
    # Pdb().set_trace()
    utils.log('start train/validate cycle')
    best_score = 0
    for epoch in range(start_epoch, num_epochs):
        this_train_sequence.compute_sequence(epoch, model, my_loss_fn, train_loader,
                              optimizer, 'train', tfh, args.backprop_batch_size, [args.lr, exp_name])

        if len(val_loader.dataset) != 0:
            rec,i = this_train_sequence.compute_sequence(epoch, model, my_loss_fn, val_loader,
                                        None, 'eval', tfh, args.backprop_batch_size, [args.lr, exp_name])
        else:
            rec,i = this_train_sequence.compute_sequence(epoch, model, my_loss_fn, train_loader,
                              None, 'eval_train', tfh, None, [args.lr, exp_name])

        is_best = False
        utils.log('best score: {}, this score: {}'.format(best_score, rec[i]))
        if rec[i] > best_score:
            best_score = rec[i]
            is_best = True
        #
        utils.save_checkpoint( {
        'epoch': epoch,
        'best_score': best_score,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'is_best': is_best
        } , epoch, is_best, checkpoint_file, best_checkpoint_file)
    #
    tfh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='exp name',
                        type=str, default='bilstm')
    
    parser.add_argument('--model', help='model  - bilstm or bilstm_crf',
                        type=str, default='bilstm')

    parser.add_argument('--train_path',
                        help='numericalized data in pickle', type=str,
                        default='/home/cse/phd/csz178057/phd/nlp/ner_tagging/data//train2.pkl')
                        #default='/home/cse/phd/csz178057/phd/nlp//ner_tagging/data/amazon13k/train1.pkl')
    parser.add_argument('--vocab_path',
                        help='vocab in pickle', type=str,
                        default='/home/cse/phd/csz178057/phd/nlp//ner_tagging/data//vocab1.pkl')
                        #default='/home/cse/phd/csz178057/phd/nlp//ner_tagging/data/amazon13k/vocab1.pkl')
    
    parser.add_argument('--output_path',
                        help='output path', type=str,
                        default='/home/cse/phd/csz178057/phd/nlp/ner_tagging/output/crafted_final')
    
    parser.add_argument('--n_fold',
                        help='max_sentence length', type=int,
                        default=10)
    
    parser.add_argument('--gid',
                        help='max_sentence length', type=int,
                        default=10)
    
    parser.add_argument('--intermediate_size',
                        help='number of hidden units', type=int,
                        default=128)
    
    parser.add_argument('--hidden_size',
                        help='number of hidden units', type=int,
                        default=128)
    
    parser.add_argument('--batch_size',
                        help='number of batch size', type=int,
                        default=256)
    
    parser.add_argument('--num_epochs',
                        help='number of epcohs for training', type=int,
                        default=38)
    
    parser.add_argument('--lr',
                        help='learning rate', type=float,
                        default=0.001)
    
    parser.add_argument('--decay',
                        help='decay in adam', type=float,
                        default=0.0)
    
    parser.add_argument('--max_length',
                        help='max_sentence length', type=int,
                        default=999999)

    parser.add_argument('--vocab_size',
                        help='max_sentence length', type=int,
                        default=9999999)

    parser.add_argument('--min_count',
                        help='max_sentence length', type=int,
                        default=1)

    parser.add_argument('--backprop_batch_size',
                        help='batch size for backprop', type = int, default=256)

    parser.add_argument('--version',
                        help='version 1 for latest', type = int, default=1)

    parser.add_argument('--error_analysis',
                        help='version 1 for latest', type = bool, default=False)
    
    parser.add_argument('--drop_prob',
                        help='unk_training', type = float, default=0.9)

    parser.add_argument('--crafted',
                        help='hand crafted features?', type = bool, default=False)

    parser.add_argument('--num_crafted',
                        help='how many hand crafted features?', type = int, default=28)

    args = parser.parse_args()
    #Pdb().set_trace()
    if args.error_analysis:
        predict(args)
    else:
        main(args)

