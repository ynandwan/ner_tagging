import pickle,os,sys
import data_helpers as tdh
import numpy as np
train_file = '../data/ner.txt'
train_output_file = '../data/train1.pkl'
vocab_output_file = '../data/vocab1.pkl'
word_count_txt_file = '../data/vocab_freq.txt'

num_features = 300
model_name = os.path.join('/home/yatin/phd/nlp/project/xmlcnn/theano_code/word2vec_models/', 'glove.6B.%dd.txt' % (num_features))
#model_name = os.path.join('/home/cse/phd/csz178057/scratch/squad/data', 'glove.6B.%dd.txt' % (num_features))
ner_file = train_file

(token_list, tag_list, raw_token_list) = utils.prepare_data(ner_file,False)

tag_dict  = {'D': 0 , 'T': 1, 'O': 2}
tag_list1 = list(map(lambda x: np.array(list(map(lambda y: tag_dict[y], x))), tag_list))

vocabs = tdh.get_vocabs_embeddings(token_list, model_name, num_features)
vocabs['y_dict'] = tag_dict
vocabs['y_dict_inv'] = dict([(tag_dict[k],k) for k in tag_dict])

x_trn = tdh.build_input_data(token_list, vocabs['vocabulary'])
pickle.dump({'x': x_trn,  'y': tag_list1}, open(train_output_file,'wb'))
pickle.dump(vocabs, open(vocab_output_file,'wb'))
print('\n'.join(['{},{}'.format(x[0],x[1]) for x in word_counts.most_common(None)]),file = open(word_count_txt_file, 'w'))
