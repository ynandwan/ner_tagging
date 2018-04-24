from datetime import datetime as dt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import torch
import shutil
import utils
from IPython.core.debugger import Pdb
from sklearn import metrics
import numpy as np


cuda = False

def stats_to_log(p, t):
    (pr, re, f, s) = metrics.precision_recall_fscore_support(t,p)
    accuracy = metrics.accuracy_score(t,p)
    macro_f_ex_normal = (f[:-1]).mean()
    rec = []
    rec.extend(list(pr))
    rec.extend(list(re))
    rec.extend(list(f))
    rec.extend(list(s))
    rec.extend([accuracy,macro_f_ex_normal])
    return rec


def compute_sequence(epoch, model, loss_fn, loader, optimizer=None, mode='eval', fh=None, backprop_batch_size=None, tolog=[],return_preds=False):
    global cuda
    #Pdb().set_trace() 
    if backprop_batch_size is None:
        backprop_batch_size = loader.batch_sampler.batch_size
    t1 = time.time()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    last_print = 0
    count = 0
    num_words  = 0
    cum_loss = 0
    
    ypred_cum = []
    y_cum = []
    #idx_mask = []
    
    #variables to write output to a file in correct order
    if return_preds:
        #all_y= [None for i in  range(len(loader.dataset))]
        #all_x = [None for i in range(len(loader.dataset))]
        all_pred = [None for i in range(len(loader.dataset))]


    if mode == 'train':
        this_backprop_count = 0
        optimizer.zero_grad()
        backprop_loss = 0

    for x, y, z, idx in loader:
        # break
        count += len(idx)
        num_words += y.shape[0]*y.shape[1]
        # print(len(idx))
        #
        volatile = True
        if mode == 'train':
            this_backprop_count += len(idx)
            volatile = False

        x, y, z  = Variable(x, volatile=volatile), Variable(
            y.long(), volatile=volatile), Variable(z.float(), volatile = volatile)
        if cuda:
            x, y, z  = x.cuda(), y.cuda(), z.cuda()
        #
        loss,pred = loss_fn(x,y,z,model)
        #ypred = model(x)
         #loss = criterion(ypred.transpose(1,2), y)
        if mode == 'train':
            backprop_loss += loss
            if this_backprop_count >= backprop_batch_size:
                #utils.log("backproping now: {0}".format(this_backprop_count))
                backprop_loss.backward()
                # loss.backward()
                optimizer.step()
                this_backprop_count = 0
                backprop_loss = 0
                optimizer.zero_grad()
        #

        y = y.data.cpu().numpy()
        x = x.data.cpu().numpy()
        pred = pred.cpu().numpy()
        y_cum.extend(y.flatten())
        ypred_cum.extend(pred.flatten())
        cum_loss = cum_loss + loss.data[0]*y.shape[0]*y.shape[1]

        if return_preds:    
            for i in range(len(idx)):
                #all_y[idx[i]] = y[i,:]
                all_pred[idx[i]] = pred[i,:]
                #all_x[idx[i]] = x[i,:]


        

        if (count - last_print) >= 20000:
            last_print = count
            rec = [epoch, mode, 1.0 * cum_loss / num_words,
                    count, num_words, time.time() - t1] + tolog + stats_to_log(ypred_cum, y_cum)

            utils.log(','.join([str(round(x, 5)) if isinstance(
                x, float) else str(x) for x in rec]))

    
    rec = [epoch, mode, 1.0 * cum_loss / num_words,
                    count, num_words, time.time() - t1] + tolog + stats_to_log(ypred_cum, y_cum)

    utils.log(','.join([str(round(x, 5)) if isinstance(
        x, float) else str(x) for x in rec]), file=fh)

    if return_preds:
        return(rec,-1,all_pred)
    else:
        return (rec,-1)
