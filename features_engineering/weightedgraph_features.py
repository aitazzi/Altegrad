import networkx as nx
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from networkx.exception import NetworkXNoPath

def generate_weightedgraph_features(path, manual_cv=False):
	"""
	Generate weighted graph features for question pairs data. 
	Features will be written in a csv file in path folder.

	Args:
	    path: folder containing train.csv and test.csv and to write csv features file.

	Return:

	"""

	# Load training and test set
	train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	# Drop useless columns
	train = train.drop(['id','question1','question2'], axis=1)
	test = test.drop(['id','question1','question2'], axis=1)

	# Change train et test
	if manual_cv is True:
	    test1 = test.copy()
	    test2 = train.copy()[60000:]
	    test2 = test2.drop(['is_duplicate'], axis=1)
	    test = pd.concat([test2, test1], axis=0)
	    train = train.copy()[:60000]

	# Initialize graph
	G=nx.Graph()
	G.add_nodes_from(pd.concat([train['qid1'], train['qid2'], test['qid1'], test['qid2']], axis=0).unique())

	# Edges for is_duplicate=1
	edge_list = []
	for index, row in train[train['is_duplicate']==1].iterrows():
	        edge_list.append([train['qid1'][index],train['qid2'][index],1])
	G.add_weighted_edges_from(edge_list)

	# Edges for is_duplicate=0
	edge_list = []
	for index, row in train[train['is_duplicate']==0].iterrows():
	        edge_list.append([train['qid1'][index],train['qid2'][index],20])
	G.add_weighted_edges_from(edge_list)

	# Edges for test set (no informations of is_duplicate)
	edge_list = []
	for index, row in test.iterrows():
	        edge_list.append([test['qid1'][index],test['qid2'][index],10])
	G.add_weighted_edges_from(edge_list)

	print('Number of nodes:', G.number_of_nodes())
	print('Number of edges:', G.number_of_edges())

	# Computing train features
	print('Computing train features')
	for index, row in tqdm(train.iterrows()):

		# Remove the edge between the pair of question
	    G.remove_edge(train['qid1'][index],train['qid2'][index])

	    # Compute shortest path
	    try:
	        train.loc[index,'shortest_path_weighted'] = nx.dijkstra_path_length(G, train['qid1'][index], train['qid2'][index])
	    except NetworkXNoPath:
	        train.loc[index,'shortest_path_weighted'] = 50

	    # Reset the deleted edge
	    if train['is_duplicate'][index] == 1:
	        G.add_weighted_edges_from([[train['qid1'][index],train['qid2'][index],1]])
	    elif train['is_duplicate'][index] == 0:
	        G.add_weighted_edges_from([[train['qid1'][index],train['qid2'][index],20]])

	# Drop the useless columns
	train = train.drop(['qid1','qid2','is_duplicate'],axis=1)

	print('Computing test features')
	for index, row in tqdm(test.iterrows()):

		# Remove the edge between the pair of question
	    G.remove_edge(test['qid1'][index],test['qid2'][index])

	    # Compute shortest path
	    try:
	        test.loc[index,'shortest_path_weighted'] = nx.dijkstra_path_length(G, test['qid1'][index], test['qid2'][index])
	    except NetworkXNoPath:
	        test.loc[index,'shortest_path_weighted'] = 50
	    
	    # Reset the deleted edge
	    G.add_weighted_edges_from([[test['qid1'][index],test['qid2'][index],10]])

	test = test.drop(['qid1','qid2'],axis=1)

	# Rechange dataset
	if manual_cv is True:
	    train1 = train.copy()
	    train2 = test.copy()[:20100]
	    train_fin = pd.concat([train1, train2], axis=0)
	    test_fin = test.copy()[20100:]
	    train=train_fin.copy()
	    test=test_fin.copy()

	print('Writing train features...')
	train.to_csv(os.path.join(path,'train_weightedgraph_feat.csv'))

	print('Writing test features...')
	test.to_csv(os.path.join(path,'test_weightedgraph_feat.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_weightedgraph_feat.csv")