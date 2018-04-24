#ref - https://github.com/threelittlemonkeys/lstm-crf-pytorch/blob/master/model.py
#ref - http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import numpy as np
from IPython.core.debugger import Pdb
cuda = False
LOW_POT = -10000.0

class BilstmCRFSequenceTagger1(nn.Module):
    def __init__(self,num_labels, vocab_size, embedding_size  = 300,hidden_size = 256, intermediate_size = 128, embedding_init = None): 
        super(BilstmCRFSequenceTagger1, self).__init__()

        self.bilstm = BilstmSequenceTagger(num_labels, vocab_size, embedding_size, hidden_size, intermediate_size, embedding_init)
        self.crf = CRF(num_labels)

        if cuda:
            self = self.cuda()

    # convention  - forward always give predictions. hence run viterbi
    def forward(self,x):
        unary_pot = self.bilstm(x)
        unary_pot = F.pad(unary_pot,(0,2),'constant',LOW_POT)
        return self.crf.forward(unary_pot)
        #return fp, path_score

    def neg_log_likelihood_loss(self,x,y):
        unary_pot = self.bilstm(x)
        unary_pot = F.pad(unary_pot,(0,2),'constant',LOW_POT)
        partition = self.crf.partition(unary_pot)
        gold_score = self.crf.score(unary_pot,y)
        return partition - gold_score 

    def gold_viterbi_loss(self,x,y):
        unary_pot = self.bilstm(x)
        unary_pot = F.pad(unary_pot,(0,2),'constant',LOW_POT)
        gold_score = self.crf.score(unary_pot,y)
        viterbi_score,_ = self.crf(unary_pot)
        loss,_= torch.max(torch.stack([Variable(utils.zeros(gold_score.size(0))), viterbi_score - gold_score],dim=1),dim = 1)
        return loss

class BilstmCRFSequenceTagger(nn.Module):
    def __init__(self,num_labels, vocab_size, embedding_size  = 300,hidden_size = 256, intermediate_size = 128, embedding_init = None): 
        super(BilstmCRFSequenceTagger, self).__init__()

        self.bilstm = BilstmSequenceTagger(num_labels+2, vocab_size, embedding_size, hidden_size, intermediate_size, embedding_init)
        self.crf = CRF(num_labels)

        if cuda:
            self = self.cuda()

    # convention  - forward always give predictions. hence run viterbi
    def forward(self,x):
        unary_pot = self.bilstm(x)
        return self.crf.forward(unary_pot)
        #return fp, path_score

    def neg_log_likelihood_loss(self,x,y):
        unary_pot = self.bilstm(x)
        partition = self.crf.partition(unary_pot)
        gold_score = self.crf.score(unary_pot,y)
        return partition - gold_score 

    def gold_viterbi_loss(self,x,y):
        unary_pot = self.bilstm(x)
        gold_score = self.crf.score(unary_pot,y)
        viterbi_score,_ = self.crf(unary_pot)
        loss,_= torch.max(torch.stack([Variable(utils.zeros(gold_score.size(0))), viterbi_score - gold_score],dim=1),dim = 1)
        return loss


class CRF(nn.Module):
    def __init__(self, num_labels):
        super(CRF, self).__init__()
        self.num_labels = num_labels
        self.total_labels = self.num_labels + 2 # start and end
        self.START_IDX = self.num_labels
        self.END_IDX = self.START_IDX + 1
        #i,j = from j to i
        self.transition_table = nn.Parameter(torch.randn(self.total_labels, self.total_labels))
        self.transition_table.data[self.START_IDX,:] = LOW_POT
        self.transition_table.data[:,self.END_IDX] = LOW_POT
        if cuda:
            self = self.cuda()

    # to get partition function. Called while training to compute loss
    def partition(self,unary_pot):
        score = utils.Tensor(unary_pot.size()[0], self.total_labels).fill_(LOW_POT)
        score[:, self.START_IDX] = 0.0
        score = Variable(score)
        for t in range(unary_pot.size(1)): # iterate through the sequence
            score_t = score.unsqueeze(-1).expand(-1,-1,self.total_labels)
            emit = unary_pot[:, t,:].unsqueeze(-1).expand(-1,-1,self.total_labels).transpose(1,2)
            trans = self.transition_table.unsqueeze(0).expand(unary_pot.size()[0],-1,-1).transpose(1,2)
            score = utils.log_sum_exp(score_t + emit + trans,1)
        #
        #take care of transition to self.END_IDX
        score = score + self.transition_table[self.END_IDX].unsqueeze(0).expand_as(score)
        score = utils.log_sum_exp(score)
        return score # partition function        

    #viterbi decode - convention - always call forward for predictions.
    def forward(self,unary_pot):
        score = utils.Tensor(unary_pot.size()[0], self.total_labels).fill_(LOW_POT)
        score[:, self.START_IDX] = 0.0
        score = Variable(score)
        back_pointers = []
        for t in range(unary_pot.size(1)): # iterate through the sequence
            score_t = score.unsqueeze(-1).expand(-1,-1,self.total_labels)
            trans = self.transition_table.unsqueeze(0).expand(unary_pot.size()[0],-1,-1).transpose(1,2)
            score = score_t + trans
            score,this_back_pointer = torch.max(score,1)
            score = score + unary_pot[:,t,:] 
            back_pointers.append(this_back_pointer)
        #
        score = score + self.transition_table[self.END_IDX].unsqueeze(0).expand_as(score)
        path_score,last_back_pointer =  torch.max(score,1)
        pointers = [last_back_pointer]
        for t in range(unary_pot.size(1)-1,-1,-1):
            last_back_pointer = back_pointers[t][np.arange(unary_pot.size(0)),last_back_pointer]
            pointers.append(last_back_pointer)
        
        must_be_start = pointers.pop()
        assert (must_be_start == self.START_IDX).all()
        bp = torch.stack(pointers,1)
        inv_idx = Variable(utils.Tensor(np.arange(bp.size(1)-1,-1,-1)).long())
        fp = bp.index_select(1, inv_idx)
        return path_score, fp

    def score(self,unary_pot, fp):
        score = Variable(utils.Tensor(unary_pot.size(0)).fill_(0.0))
        tag_seq = torch.cat([Variable(utils.Tensor(unary_pot.size(0),1).long().fill_(self.START_IDX)),fp],1)
        for t in range(unary_pot.size(1)): # iterate on sequence
            emit = torch.cat([unary_pot[b, t, tag_seq[b, t + 1].data[0]] for b in range(unary_pot.size(0))])
            trans = torch.cat([self.transition_table[seq[t + 1].data[0], seq[t].data[0]] for seq in tag_seq])
            score = score + emit + trans
        #
        score = score + torch.cat([self.transition_table[self.END_IDX, seq[-1].data[0]] for seq in tag_seq])

        return score            
            

class BilstmSequenceTagger(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, hidden_size = 256, intermediate_size = 128, embedding_init = None):
        super(BilstmSequenceTagger, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size,embedding_size)
        self.intermediate_size = intermediate_size
        if embedding_init is not None:
            self.word_embeddings.weight.data.copy_(torch.FloatTensor(embedding_init))

        #  
        self.embedding_dropout = nn.Dropout(p = 0.25)
    
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional = True, batch_first= True, dropout = 0.5)

        self.fc1 = nn.Sequential(
                 nn.Linear(2*self.hidden_size, self.intermediate_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )

        self.classifier = nn.Linear(self.intermediate_size, self.num_labels)
        if cuda: 
            self = self.cuda()

    def forward(self, x):
        emb = self.word_embeddings(x)
        emb = self.embedding_dropout(emb)
        h,_ = self.rnn(emb,None) 
        h = self.fc1(h)
        h = self.classifier(h)
        return h


class BilstmSequenceTaggerCraftedFeatures(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, hidden_size = 256, intermediate_size = 128, embedding_init = None, crafted_features_size=50, crafted_embedding_size=100):
        super(BilstmSequenceTaggerCraftedFeatures, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size,embedding_size)

        self.crafted_embeddings = nn.Sequential(
                nn.Linear(crafted_features_size, crafted_embedding_size), 
                nn.ReLU(),
                nn.Dropout(0.25)
                )

        self.intermediate_size = intermediate_size
        if embedding_init is not None:
            self.word_embeddings.weight.data.copy_(torch.FloatTensor(embedding_init))

        #  
        self.embedding_dropout = nn.Dropout(p = 0.25)
    
        self.rnn = nn.LSTM(self.embedding_size+crafted_embedding_size, self.hidden_size, bidirectional = True, batch_first= True, dropout = 0.5)

        self.fc1 = nn.Sequential(
                 nn.Linear(2*self.hidden_size, self.intermediate_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )
        #
        self.classifier = nn.Linear(self.intermediate_size, self.num_labels)
        if cuda: 
            self = self.cuda()

    def forward(self, x, crafted_features):
        emb = self.word_embeddings(x)
        emb = self.embedding_dropout(emb)
        ce = self.crafted_embeddings(crafted_features)
        #Pdb().set_trace()
        emb1 = torch.cat([emb,ce], dim =-1)
        h,_ = self.rnn(emb1,None) 
        h = self.fc1(h)
        h = self.classifier(h)
        return h

