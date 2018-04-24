import pandas as pd
import os,sys, shutil, glob
import numpy as np

#exp_names = ['exp_kernels','exp_num_features_correct_bs']
#flist = glob.glob('../output/eurlex/*/*.csv')
dirlist= glob.glob('../output/crafted_gz_lt')
olist = []
datlist = []
for di in dirlist:
    flist = glob.glob(di+'/*.csv')
    #flist = glob.glob('../output/mybicrf_mn_bilstm_crf*.csv')
    colnames = ['dt','epoch','mo','loss','ns','nw','t','lr','ex','p1','p2','p3','r1','r2','r3','f1','f2','f3','s1','s2','s3','a','mf']
    
    
    dat = None
    for f in flist:
        print(f)
        a = pd.read_csv(f,header=None, names = colnames, index_col=False)
        if dat is None:
            dat = a
        else:
            dat = dat.append(a)
    
    dat =dat.reset_index()
    del dat['index']
    
    dat['ex'] =dat['ex'].apply(lambda x: x.replace('bilstm_crf','bilstmcrf'))
    dat['params'] = dat.ex.apply(lambda x: ':'.join(x.split('_')[0::2]))
    dat['params1'] = dat['params'].apply(lambda x: ':'.join(x.split(':')[:-1]))
    dat['gid'] = dat['params'].apply(lambda x: x.split(':')[-1])
    
    
    #a = dat.groupby(['params1','gid','mo']).agg({'mf':'max'})
    
    a  = dat.loc[dat.groupby(["params1", "gid","mo"])["mf"].idxmax()]
    a = a.reset_index()
    del a['index']
    
    
    #dat.groupby(["params1", "gid","mo"])["mf"].idxmax().values
    #a = a.reset_index()
    
    b= a.groupby(['params1','mo']).agg('mean')
    olist.append(b)
    datlist.append(dat)
    #b.transpose()
    #def get_row_for_max_mf(x):
#    x = x.reset_index()
#    return(x[x['mf'].idxmax()])
#a = dat.groupby(['params1','gid','mo']).apply(get_row_for_max_mf)



c = dat.groupby(['mo','epoch']).agg('mean')
c = c.reset_index()
c.head()
c[c.mo == 'eval']['mf']
c[c.mo == 'eval']['mf'].max()
c[c.mo == 'eval']['mf'].idxmax()


