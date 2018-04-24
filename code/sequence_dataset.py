import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy
from IPython.core.debugger import Pdb 
import pickle


MAX_SENTENCE_LENGTH = 999999

def assign_groups_for_cv(ds, num_groups):
    np.random.seed(25120)
    gid = np.random.randint(num_groups, size = len(ds))
    return gid

def get_train_val_split(ds, gid_list, gid_for_val, drop_words = False, drop_prob = 0.95, crafted=False):
    np.random.seed(25120+17*gid_for_val+11)
    train_ds = copy.deepcopy(ds)
    val_ds = copy.deepcopy(ds)
    train_ds.data['x'] = [ds.data['x'][i] for i in range(len(ds)) if gid_list[i] != gid_for_val]
    train_ds.data['y'] = [ds.data['y'][i] for i in range(len(ds)) if gid_list[i] != gid_for_val]
    val_ds.data['x'] = [ds.data['x'][i] for i in range(len(ds)) if gid_list[i] == gid_for_val]
    val_ds.data['y'] = [ds.data['y'][i] for i in range(len(ds)) if gid_list[i] == gid_for_val]
    if crafted:
        train_ds.data['z'] = [ds.data['z'][i] for i in range(len(ds)) if gid_list[i] != gid_for_val]
        val_ds.data['z'] = [ds.data['z'][i] for i in range(len(ds)) if gid_list[i] == gid_for_val]

    if drop_words:
        for i in range(len(train_ds.data['x'])):
            temp = np.random.random_sample(len(train_ds.data['x'][i]))
            train_ds.data['x'][i] = np.array([train_ds.data['x'][i][j] if temp[j] < drop_prob else 0 for j in range(len(temp))])    #   
    return (train_ds, val_ds)


class sequence_dataset(torch.utils.data.Dataset):
    vocabulary_inv = None
    vocab_size = None
    def __init__(self, root_dir, train_val_test= 'train', pkl_data_path = None, max_length = None, vocab_size = 500000, min_count = 1, word_counts = None, vocabulary_inv = None, crafted_features = False):
        self.root_dir=  root_dir
        self.crafted_features = crafted_features
        if isinstance(pkl_data_path,dict):
            self.data = pkl_data_path
        else:
            self.data = pickle.load(open(os.path.join(root_dir,pkl_data_path),'rb'))
        #
        if max_length is None:
            self.max_length = MAX_SENTENCE_LENGTH
        else:
            self.max_length = max_length
        #
        if sequence_dataset.vocabulary_inv is None:
            assert(vocabulary_inv is not None)
            vocab_size = min(vocab_size, len(vocabulary_inv)) + 2
            sequence_dataset.vocabulary_inv = vocabulary_inv[:vocab_size]
            sequence_dataset.vocabulary_inv = vocabulary_inv[:2] + [x for x in sequence_dataset.vocabulary_inv[2:] if word_counts[x] >= min_count]
            sequence_dataset.vocab_size = len(sequence_dataset.vocabulary_inv)

        #
        for i in range(len(self.data['x'])):
            self.data['x'][i] = np.array(self.data['x'][i])
            self.data['x'][i][self.data['x'][i] >= sequence_dataset.vocab_size] = 0
            n = min(len(self.data['x'][i]),self.max_length)
            self.data['x'][i] = self.data['x'][i][:n]
            self.data['z'][i] = self.data['z'][i][:n]

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self,idx):
        if self.crafted_features:
            return (self.data['x'][idx], self.data['y'][idx], self.data['z'][idx], idx)
        else:
            return (self.data['x'][idx], self.data['y'][idx], idx)
        #return (self.data['x'][idx], np.ravel(self.data['y'][idx].todense()), idx)



