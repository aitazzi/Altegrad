import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import scipy.stats as sps
import os


def generate_kcores(path):
    seed = 1024
    np.random.seed(seed)

    train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","label"])
    test = pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

    data_all = pd.concat([train, test])[['question1','question2']]

    #dup index
    q_all = pd.DataFrame(np.hstack([train['question1'], test['question1'],
                       train['question2'], test['question2']]), columns=['question'])
    q_all = pd.DataFrame(q_all.question.value_counts()).reset_index()

    q_num = dict(q_all.values)
    q_index = {}
    for i,key in enumerate(q_num.keys()):
        q_index[key] = i
    data_all['q1_index'] = data_all['question1'].map(q_index)
    data_all['q2_index'] = data_all['question2'].map(q_index)


    #link edges
    q_list = {}
    dd = data_all[['q1_index','q2_index']].values
    for i in tqdm(np.arange(data_all.shape[0])):
    #for i in np.arange(dd.shape[0]):
        q1,q2=dd[i]
        if q_list.setdefault(q1,[q2])!=[q2]:
            q_list[q1].append(q2)
        if q_list.setdefault(q2,[q1])!=[q1]:
            q_list[q2].append(q1)


    common_fea = np.zeros((data_all.shape[0],3))
    for i in tqdm(np.arange(data_all.shape[0])):
        q1,q2 = dd[i]
        if (q1 not in q_list)|(q2 not in q_list):
            continue
        nei_q1 = set(q_list[q1])
        nei_q2 = set(q_list[q2])

        f_1 = len(nei_q1.intersection(nei_q2))
        common_fea[i][0] = f_1
        common_fea[i][1] = len(nei_q1)
        common_fea[i][2] = len(nei_q2)

    train_common = common_fea[:train.shape[0]]
    test_common = common_fea[train.shape[0]:]

    # pd.to_pickle(train_common,'data/train_neigh.pkl')
    # pd.to_pickle(test_common,'data/test_neigh.pkl')
    train_cores=pd.DataFrame(train_common, columns=['core1','core2','core3'])
    test_cores=pd.DataFrame(test_common, columns=['core1','core2','core3'])


    
    print('Writing train features...')
    train_cores.to_csv(os.path.join(path,'train_kcores.csv'))

    print('Writing test features...')
    test_cores.to_csv(os.path.join(path,'test_kcores.csv'))

    print('CSV written ! see: ', path, " | suffix: ", "_kcores.csv")



