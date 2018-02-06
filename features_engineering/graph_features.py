import networkx as nx
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from networkx.exception import NetworkXNoPath


def generate_graph_features(path):
	"""
	Generate graph features for question pairs data. 
	Features will be written in a csv file in path folder.

	Args:
	    path: folder containing train.csv and test.csv and to write csv features file.

	Return:

	"""

	# Load training and test set
	train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	# Drop useless columns
	train = train.drop(['id','question1','question2', 'is_duplicate'], axis=1)
	test = test.drop(['id','question1','question2'], axis=1)

	train_test = pd.concat([train,test], ignore_index=True)

	# Initialize graph
	G=nx.Graph()

	# Create edges
	edge_list = []
	for index, row in train_test.iterrows():
	    edge_list.append([train_test['qid1'][index],train_test['qid2'][index]])
	G.add_edges_from(edge_list)

	print('Number of nodes:', G.number_of_nodes())
	print('Number of edges:', G.number_of_edges())

	# Computing train features
	print('Computing train features')
	for index, row in tqdm(train.iterrows()):

		# Generate neighbors of each questions
	    neigh_1 = G.neighbors(train['qid1'][index])
	    neigh_2 = G.neighbors(train['qid2'][index])

	    # Number of neigbors
	    train.loc[index,'q1_neigh'] = len(neigh_1)
	    train.loc[index,'q2_neigh'] = len(neigh_2)

	    # Count the common and the distinct neighbors
	    train.loc[index,'common_neigh'] = len(list(nx.common_neighbors(G,train['qid1'][index],train['qid2'][index])))
	    train.loc[index,'distinct_neigh'] = len(neigh_1)+len(neigh_2)-len(list(nx.common_neighbors(G,train['qid1'][index],train['qid2'][index])))

	    # Compute the clique size
	    train.loc[index,'clique_size'] = nx.node_clique_number(G,train['qid1'][index])

	    # Generate shortest path
	    # Cut the edge to compute features
	    G.remove_edge(train['qid1'][index],train['qid2'][index])
	    # Compute shortest path
	    try:
	        train.loc[index,'shortest_path'] = nx.shortest_path_length(G, train['qid1'][index], train['qid2'][index])
	    except NetworkXNoPath:
	        train.loc[index,'shortest_path'] = 10
		# Reset the edge
	    G.add_edge(train['qid1'][index],train['qid2'][index])

	# Drop the useless columns
	train = train.drop(['qid1','qid2'],axis=1)

	print('Writing train features...')
	train.to_csv(os.path.join(path,'train_graph_feat.csv'))

	print('Computing test features')
	for index, row in tqdm(test.iterrows()):

		# Generate neighbors of each questions
	    neigh_1 = G.neighbors(test['qid1'][index])
	    neigh_2 = G.neighbors(test['qid2'][index])

	    # Number of neigbors
	    test.loc[index,'q1_neigh'] = len(neigh_1)
	    test.loc[index,'q2_neigh'] = len(neigh_2)

	    # Count the common and the distinct neighbors
	    test.loc[index,'common_neigh'] = len(list(nx.common_neighbors(G,test['qid1'][index],test['qid2'][index])))
	    test.loc[index,'distinct_neigh'] = len(neigh_1)+len(neigh_2)-len(list(nx.common_neighbors(G,test['qid1'][index],test['qid2'][index])))

	    # Compute the clique size
	    test.loc[index,'clique_size'] = nx.node_clique_number(G,test['qid1'][index])

	    # Generate shortest path
	    # Cut the edge to compute features
	    G.remove_edge(test['qid1'][index],test['qid2'][index])
	    # Compute shortest path
	    try:
	        test.loc[index,'shortest_path'] = nx.shortest_path_length(G, test['qid1'][index], test['qid2'][index])
	    except NetworkXNoPath:
	        test.loc[index,'shortest_path'] = 10
    	# Reset the edge
	    G.add_edge(test['qid1'][index],test['qid2'][index])

	# Drop the useless columns
	test = test.drop(['qid1','qid2'],axis=1)

	print('Writing test features...')    
	test.to_csv(os.path.join(path,'test_graph_feat.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_graph_feat.csv")