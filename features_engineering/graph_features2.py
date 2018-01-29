import networkx as nx
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from networkx.exception import NetworkXNoPath

def generate_grapf_features2(path):
	train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	train = train.drop(['id','question1','question2'], axis=1)
	test = test.drop(['id','question1','question2'], axis=1)


	G=nx.Graph()
	G.add_nodes_from(pd.concat([train['qid1'], train['qid2'], test['qid1'], test['qid2']], axis=0).unique())

	edge_list = []
	for index, row in train[train['is_duplicate']==1].iterrows():
	        edge_list.append([train['qid1'][index],train['qid2'][index]])
	G.add_edges_from(edge_list)

	print('Number of nodes:', G.number_of_nodes())
	print('Number of edges:', G.number_of_edges())

	# Computing train features
	print('Computing train features')
	for index, row in tqdm(train.iterrows()):
	    try:
	        train.loc[index,'shortest_path'] = nx.shortest_path_length(G, train['qid1'][index], train['qid2'][index])
	    except NetworkXNoPath:
	        train.loc[index,'shortest_path'] = 10
	    
	train = train.drop(['qid1','qid2','is_duplicate'],axis=1)
	print('Writing train features...')
	train.to_csv(os.path.join(path,'train_graph_feat2.csv'))


	print('Computing test features')
	for index, row in tqdm(test.iterrows()):
	    try:
	        test.loc[index,'shortest_path'] = nx.shortest_path_length(G, test['qid1'][index], test['qid2'][index])
	    except NetworkXNoPath:
	        test.loc[index,'shortest_path'] = 10

	test = test.drop(['qid1','qid2'],axis=1)
	print('Writing test features...')
	test.to_csv(os.path.join(path,'test_graph_feat2.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_graph_feat2.csv")