import networkx as nx
import pandas as pd
from tqdm import tqdm
import os


def generate_question_kcores(path):
	"""
	Generate weighted graph features based on k-cores for Quora question data:
	largest subgraph where every node is connected to at least “k” nodes for each question of the pair,
	ratio, difference and normed difference.
	Features will be written in a csv file in path folder.

	Args:
	    path: folder containing train.csv and test.csv and to write csv features file.

	Return:

	"""

	# Load training and test set
	df_train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","label"])
	df_test = pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	dfs = (df_train, df_test)

	# Preprocessing the questions to put them as nodes in the graph
	questions = []
	for df in dfs:
	    df['question1'] = df['question1'].str.lower()
	    df['question2'] = df['question2'].str.lower()
	    questions += df['question1'].tolist()
	    questions += df['question2'].tolist()

	# Initialize graph    
	graph = nx.Graph()
	graph.add_nodes_from(questions)

	# Add edges
	for df in [df_train, df_test]:
	    edges = list(df[['question1', 'question2']].to_records(index=False))
	    graph.add_edges_from(edges)
	graph.remove_edges_from(graph.selfloop_edges())

	df = pd.DataFrame(data=graph.nodes(), columns=["question"])
	df['kcores'] = 1

	# Store the argest subgraph where every node is connected to at least 'k' nodes
	# Note that k is bounded by 30 to avoid too long computational time.
	n_cores = 30
	for k in tqdm(range(2, n_cores + 1)):
	    ck = nx.k_core(graph, k=k).nodes()
	    df['kcores'][df.question.isin(ck)] = k

	# Store intermediate result     
	df.to_csv(os.path.join(path,"question_kcores.csv"), index=None)

	# Load training and test set
	df_train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","label"])
	df_test = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	dfs = (df_train, df_test)
	for df in dfs:
	    df['question1'] = df['question1'].str.lower()
	    df['question2'] = df['question2'].str.lower()

	    # Load intermediate results
	    q_kcores = pd.read_csv('data/question_kcores.csv', encoding="ISO-8859-1")
	    
	    # Store argest subgraph where every node is connected to at least “k” nodes
	    # for question 1
	    q_kcores['question1'] = q_kcores['question']
	    del q_kcores['question']
	    df['q1_kcores'] = df.merge(q_kcores, 'left')['kcores']
	    
	    # for question 2
	    q_kcores['question2'] = q_kcores['question1']
	    del q_kcores['question1']
	    df['q2_kcores'] = df.merge(q_kcores, 'left')['kcores']
	    
	    # Ratio (must be between 0 and 1)
	    df['q1_q2_kcores_ratio'] = (df['q1_kcores'] / df['q2_kcores']).apply(lambda x: x if x < 1. else 1./x)
	    # Difference
	    df['q1_q2_kcores_diff'] = (df['q1_kcores'] - df['q2_kcores']).apply(abs)
	    # Normed difference
	    df['q1_q2_kcores_diff_normed'] = (df['q1_kcores'] - df['q2_kcores']).apply(abs) / (df['q1_kcores'] + df['q2_kcores'])

	df_train, df_test = dfs
	df_train = df_train[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 'q1_q2_kcores_diff_normed']].astype("float64")
	df_test = df_test[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 'q1_q2_kcores_diff_normed']].astype("float64")


	print('Writing train features...')
	df_train.to_csv(os.path.join(path,'train_question_kcores.csv'))

	print('Writing test features...')
	df_test.to_csv(os.path.join(path,'test_question_kcores.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_question_kcores.csv")

