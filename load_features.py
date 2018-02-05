import pandas as pd
import string

def load_features(data_dir):
	"""
	Load graph features in data_dir

	Args:
	    path: directory containing csv files with features.

	Return:
		features_train: pandas Dataframe containing all features for training set.
		features_test: pandas Dataframe containing all features for test set.
	"""

	# LOAD DATA
	# -------------------------
	# Glove features
	features_train = pd.read_csv(data_dir+'train_features_glove.csv', sep=',', encoding='latin-1')
	features_test = pd.read_csv(data_dir+'test_features_glove.csv', sep=',', encoding='latin-1')
	features_train= features_train.drop(['question1', 'question2'], axis=1)
	features_test = features_test.drop(['id','qid1','qid2','question1', 'question2'], axis=1)
	data_train = pd.read_csv(data_dir+'train.csv', sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])

	# Pagerank features
	pagerank_feats_train = pd.read_csv(data_dir+"train_pagerank.csv", sep=',')
	pagerank_feats_test = pd.read_csv(data_dir+"test_pagerank.csv", sep=',')

	# Question frequency
	train_question_freq = pd.read_csv(data_dir+'train_question_freq.csv', sep=',', index_col=0)
	test_question_freq = pd.read_csv(data_dir+'test_question_freq.csv', sep=',', index_col=0)

	# Intersection of questions
	train_question_inter= pd.read_csv(data_dir+'train_question_inter.csv', sep=',', index_col=0)
	test_question_inter = pd.read_csv(data_dir+'test_question_inter.csv', sep=',', index_col=0)

	# question K-cores
	train_question_kcores = pd.read_csv(data_dir+'train_question_kcores.csv', sep=',', index_col=0)
	test_question_kcores = pd.read_csv(data_dir+'test_question_kcores.csv', sep=',', index_col=0)

	# TF-IDF
	train_tfidf = pd.read_csv(data_dir+'train_tfidf.csv', sep=',', index_col=0)
	test_tfidf = pd.read_csv(data_dir+'test_tfidf.csv', sep=',', index_col=0)

	# Graph features
	train_graph_feat = pd.read_csv(data_dir+'train_graph_feat.csv', sep=',', index_col=0)
	test_graph_feat = pd.read_csv(data_dir+'test_graph_feat.csv', sep=',', index_col=0)

	# Bigram feature
	train_bigram_feat = pd.read_csv(data_dir+'train_2gram_feat.csv', sep=',', index_col=0)
	test_bigram_feat = pd.read_csv(data_dir+'test_2gram_feat.csv', sep=',', index_col=0)

	# 3gram feature
	train_3gram_feat = pd.read_csv(data_dir+'train_3gram_feat.csv', sep=',', index_col=0)
	test_3gram_feat = pd.read_csv(data_dir+'test_3gram_feat.csv', sep=',', index_col=0)

	# spaCy feature
	train_spacy_feat = pd.read_csv(data_dir+'train_spacy_features.csv', sep=',', index_col=0)
	test_spacy_feat = pd.read_csv(data_dir+'test_spacy_features.csv', sep=',', index_col=0)

	# Graph features2 NE PAS PRENDRE !!!
	train_weightedgraph_feat = pd.read_csv(data_dir+'train_weightedgraph_feat.csv', sep=',', index_col=0)
	test_weightedgraph_feat = pd.read_csv(data_dir+'test_weightedgraph_feat.csv', sep=',', index_col=0)

	# Word features
	train_word_feat = pd.read_csv(data_dir+'train_word_feat.csv', sep=',', index_col=0)
	test_word_feat = pd.read_csv(data_dir+'test_word_feat.csv', sep=',', index_col=0)

	# Letter count features
	train_letters_count_feat = pd.read_csv(data_dir+'train_letters_count_feat.csv', sep=',', index_col=0)
	test_letters_count_feat = pd.read_csv(data_dir+'test_letters_count_feat.csv', sep=',', index_col=0)




	# ADD FEATURES
	# -------------------------
	# Add Pagerank features
	features_train[["q1_pr","q2_pr"]]=pagerank_feats_train[["q1_pr","q2_pr"]]
	features_test[["q1_pr","q2_pr"]]=pagerank_feats_test[["q1_pr","q2_pr"]]

	# Add question frequency features
	features_train[["q1_hash","q2_hash","q1_freq","q2_freq"]]=train_question_freq[["q1_hash","q2_hash","q1_freq","q2_freq"]]
	features_test[["q1_hash","q2_hash","q1_freq","q2_freq"]]=test_question_freq[["q1_hash","q2_hash","q1_freq","q2_freq"]]

	# Add intersection of questions features
	features_train['q1_q2_intersect']=train_question_inter['q1_q2_intersect']
	features_test['q1_q2_intersect']=test_question_inter['q1_q2_intersect']

	# Add question K-cores features
	features_train[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 
	                'q1_q2_kcores_diff_normed']]=train_question_kcores[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 'q1_q2_kcores_diff_normed']]
	features_test[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 
	               'q1_q2_kcores_diff_normed']]=test_question_kcores[['q1_kcores', 'q2_kcores', 'q1_q2_kcores_ratio', 'q1_q2_kcores_diff', 'q1_q2_kcores_diff_normed']]

	# Add TF-IDF features
	features_train[['word_match','tfidf_wm','tfidf_wm_stops','jaccard','wc_diff','wc_ratio','wc_diff_unique','wc_ratio_unique','wc_diff_unq_stop','wc_ratio_unique_stop','same_start',
	 'char_diff','char_diff_unq_stop','total_unique_words','total_unq_words_stop','char_ratio']]=train_tfidf[['word_match','tfidf_wm','tfidf_wm_stops','jaccard','wc_diff','wc_ratio','wc_diff_unique','wc_ratio_unique','wc_diff_unq_stop','wc_ratio_unique_stop','same_start',
	 'char_diff','char_diff_unq_stop','total_unique_words','total_unq_words_stop','char_ratio']]
	features_test[['word_match','tfidf_wm','tfidf_wm_stops','jaccard','wc_diff','wc_ratio','wc_diff_unique','wc_ratio_unique','wc_diff_unq_stop','wc_ratio_unique_stop','same_start',
	 'char_diff','char_diff_unq_stop','total_unique_words','total_unq_words_stop','char_ratio']]=test_tfidf[['word_match','tfidf_wm','tfidf_wm_stops','jaccard','wc_diff','wc_ratio','wc_diff_unique','wc_ratio_unique','wc_diff_unq_stop','wc_ratio_unique_stop','same_start',
	 'char_diff','char_diff_unq_stop','total_unique_words','total_unq_words_stop','char_ratio']]

	# Add graph features
	features_train[['q1_neigh','q2_neigh','common_neigh', 'distinct_neigh', 'clique_size', 'shortest_path']] = train_graph_feat[['q1_neigh','q2_neigh','common_neigh', 'distinct_neigh', 'clique_size', 'shortest_path']]
	features_test[['q1_neigh','q2_neigh','common_neigh', 'distinct_neigh', 'clique_size', 'shortest_path']] = test_graph_feat[['q1_neigh','q2_neigh','common_neigh', 'distinct_neigh', 'clique_size', 'shortest_path']]

	# Add bigram features
	features_train[['bigram_coocurence','bigram_distinct','bigram_nostpwrd_coocurence','bigram_nostpwrd_distinct']] = train_bigram_feat[['bigram_coocurence','bigram_distinct','bigram_nostpwrd_coocurence','bigram_nostpwrd_distinct']]
	features_test[['bigram_coocurence','bigram_distinct','bigram_nostpwrd_coocurence','bigram_nostpwrd_distinct']] = test_bigram_feat[['bigram_coocurence','bigram_distinct','bigram_nostpwrd_coocurence','bigram_nostpwrd_distinct']]

	# Add 3gram features
	features_train[['3gram_cooccurence','3gram_distinct','3gram_nostpwrd_cooccurence','3gram_nostpwrd_distinct']] = train_3gram_feat[['3gram_cooccurence','3gram_distinct','3gram_nostpwrd_cooccurence','3gram_nostpwrd_distinct']]
	features_test[['3gram_cooccurence','3gram_distinct','3gram_nostpwrd_cooccurence','3gram_nostpwrd_distinct']] = test_3gram_feat[['3gram_cooccurence','3gram_distinct','3gram_nostpwrd_cooccurence','3gram_nostpwrd_distinct']]

	# Add spaCy features
	features_train[['spacy_similarity']] = train_spacy_feat[['spacy_similarity']]
	features_test[['spacy_similarity']] = test_spacy_feat[['spacy_similarity']]

	# Add graph features2
	features_train[['shortest_path_weighted']] = train_weightedgraph_feat[['shortest_path_weighted']]
	features_test[['shortest_path_weighted']] = test_weightedgraph_feat[['shortest_path_weighted']]

	# Add graph features
	features_train[[ 'q1_how','q2_how','how_both','q1_what','q2_what','what_both','q1_which','q2_which','which_both','q1_who','q2_who','who_both','q1_where','q2_where','where_both','q1_when','q2_when','when_both','q1_why','q2_why','why_both','caps_count_q1','caps_count_q2','diff_caps','exactly_same']]=train_word_feat[[ 'q1_how','q2_how','how_both','q1_what','q2_what','what_both','q1_which','q2_which','which_both','q1_who','q2_who','who_both','q1_where','q2_where','where_both','q1_when','q2_when','when_both','q1_why','q2_why','why_both','caps_count_q1','caps_count_q2','diff_caps','exactly_same']]
	features_test[[ 'q1_how','q2_how','how_both','q1_what','q2_what','what_both','q1_which','q2_which','which_both','q1_who','q2_who','who_both','q1_where','q2_where','where_both','q1_when','q2_when','when_both','q1_why','q2_why','why_both','caps_count_q1','caps_count_q2','diff_caps','exactly_same']]=test_word_feat[[ 'q1_how','q2_how','how_both','q1_what','q2_what','what_both','q1_which','q2_which','which_both','q1_who','q2_who','who_both','q1_where','q2_where','where_both','q1_when','q2_when','when_both','q1_why','q2_why','why_both','caps_count_q1','caps_count_q2','diff_caps','exactly_same']]

	# Add letter count features
	features_train[['num_space_q1', 'num_space_q2', 'num_word_q1', 'num_word_q2', 'num_vowels_q1', 'num_vowels_q2']] = train_letters_count_feat[['num_space_q1', 'num_space_q2', 'num_word_q1', 'num_word_q2', 'num_vowels_q1', 'num_vowels_q2']]
	features_test[['num_space_q1', 'num_space_q2', 'num_word_q1', 'num_word_q2', 'num_vowels_q1', 'num_vowels_q2']] = test_letters_count_feat[['num_space_q1', 'num_space_q2', 'num_word_q1', 'num_word_q2', 'num_vowels_q1', 'num_vowels_q2']]
	for c in list(set(string.ascii_lowercase)):
		features_train[['num_' + c + '_q1', 'num_' + c + '_q2']] = train_letters_count_feat[['num_' + c + '_q1','num_' + c + '_q2']]
		features_test[['num_' + c + '_q1', 'num_' + c + '_q2']] = test_letters_count_feat[['num_' + c + '_q1','num_' + c + '_q2']]


	return features_train, features_test, data_train