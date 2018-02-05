import numpy as np 
import pandas as pd 
from collections import defaultdict
import os


def generate_question_inter(path):
	"""
	Generate vocabulary intersection Quora question data. 
	Features will be written in a csv file in path folder.

	Args:
	    path: folder containing train.csv and test.csv and to write csv features file.

	Return:

	"""

	# Function to compute vocabulary intersection
	def q1_q2_intersect(row):
	    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

	# Load training and test set
	train_orig =pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	test_orig =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	# Concatenation
	ques = pd.concat([train_orig[['question1', 'question2']], test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')


	q_dict = defaultdict(set)
	for i in range(ques.shape[0]):
	    q_dict[ques.question1[i]].add(ques.question2[i])
	    q_dict[ques.question2[i]].add(ques.question1[i])

	# Compute vocabulary intersection
	train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
	test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

	train_feat = train_orig[['q1_q2_intersect']]
	test_feat = test_orig[['q1_q2_intersect']]

	print('Writing train features...')
	train_feat.to_csv(os.path.join(path,'train_question_inter.csv'))

	print('Writing test features...')  
	test_feat.to_csv(os.path.join(path,'test_question_inter.csv'))

	print('CSV written ! see: ', path, " | suffix: ", "_question_inter.csv")