import networkx as nx
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def generate_graph_features(path):

	train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	train = train.drop(['id','question1','question2', 'is_duplicate'], axis=1)
	test = test.drop(['id','question1','question2'], axis=1)

	train_test = pd.concat([train,test], ignore_index=True)

	# Create graph
	G=nx.Graph()

	edge_list = []
	for index, row in train_test.iterrows():
		edge_list.append([train_test['qid1'][index],train_test['qid2'][index]])

	G.add_edges_from(edge_list)

	print('Number of nodes:', G.number_of_nodes())
	print('Number of edges:', G.number_of_edges())


	train['q1_neigh'] = np.nan; train['q2_neigh'] = np.nan
	train['common_neigh'] = np.nan; train['distinct_neigh'] = np.nan
	#train['all_simple_paths_3'] =  np.nan #TOO LONG !
	train['clique_size'] = np.nan
	#train['number_of_clique'] = np.nan #TOO LONG !


	# Computing train features
	print('Computing train features')
	for index, row in tqdm(train.iterrows()):
	    neigh_1 = G.neighbors(train['qid1'][index])
	    neigh_2 = G.neighbors(train['qid2'][index])
	    
	    train.loc[index,'q1_neigh'] = len(neigh_1)
	    train.loc[index,'q2_neigh'] = len(neigh_2)
	    train.loc[index,'common_neigh'] = len(list(nx.common_neighbors(G,train['qid1'][index],train['qid2'][index])))
	    train.loc[index,'distinct_neigh'] = len(neigh_1)+len(neigh_2)-len(list(nx.common_neighbors(G,train['qid1'][index],train['qid2'][index])))
	    
	    #train.loc[index,'all_simple_paths_3'] = len(list(nx.all_simple_paths(G,train['qid1'][index],train['qid2'][index])))
	    
	    train.loc[index,'clique_size'] = nx.node_clique_number(G,train['qid1'][index])
	    #train.loc[index,'number_of_clique'] = nx.number_of_cliques(G,train['qid1'][index])

	train.drop(['qid1','qid2'],axis=1)

	print('Writing train features...')	    
	train.to_csv(os.path.join(path,'train_graph_feat.csv'))

	print('Computing test features')
	for index, row in tqdm(test.iterrows()):
	    neigh_1 = G.neighbors(test['qid1'][index])
	    neigh_2 = G.neighbors(test['qid2'][index])
	    
	    test.loc[index,'q1_neigh'] = len(neigh_1)
	    test.loc[index,'q2_neigh'] = len(neigh_2)
	    test.loc[index,'common_neigh'] = len(list(nx.common_neighbors(G,test['qid1'][index],test['qid2'][index])))
	    test.loc[index,'distinct_neigh'] = len(neigh_1)+len(neigh_2)-len(list(nx.common_neighbors(G,test['qid1'][index],test['qid2'][index])))
	    
	    #test.loc[index,'all_simple_paths_3'] = len(list(nx.all_simple_paths(G,test['qid1'][index],test['qid2'][index])))
	    
	    test.loc[index,'clique_size'] = nx.node_clique_number(G,test['qid1'][index])
	    #test.loc[index,'number_of_clique'] = nx.number_of_cliques(G,test['qid1'][index])

	test.drop(['qid1','qid2'],axis=1)

	print('Writing test features...')	    
	test.to_csv(os.path.join(path,'test_graph_feat.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_graph_feat.csv")



