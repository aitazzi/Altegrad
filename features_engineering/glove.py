from gensim.models import KeyedVectors
import pandas as pd


def generate_glove_features(path, word2vec_filepath, googlenews_filepath):
	"""
    Generate glove features for Quora questions data. 
    Features will be written in a csv file in path folder.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.
        word2vec_filepath: path to word2vec file.
        googlenews_filepath: path to Google News file.

    Return:
        
    """

	# WMD distance
	def wmd(s1, s2):
	    s1 = str(s1).lower().split()
	    s2 = str(s2).lower().split()
	    stop_words = stopwords.words('english')
	    s1 = [w for w in s1 if w not in stop_words]
	    s2 = [w for w in s2 if w not in stop_words]
	    return model.wmdistance(s1, s2)

    # Sentence embedding
	def sent2vec(s):
		words = str(s).lower()
		words = word_tokenize(words)
		words = [w for w in words if not w in stop_words]
		words = [w for w in words if w.isalpha()]
		M = []
		for w in words:
		    try:
		        M.append(model[w])
		    except:
		        continue
		M = np.array(M)
		v = M.sum(axis=0)
		return v / np.sqrt((v ** 2).sum())


    # Import embedding model
	word_embedding_model_glove = KeyedVectors.load_word2vec_format(word2vec_filepath, binary=False)
	model = gensim.models.KeyedVectors.load_word2vec_format(googlenews_filepath, binary=True)

    # Load training and test set
	data_train = pd.read_csv('data/train.csv', sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
	data_train = data_train.drop(['id', 'qid1', 'qid2'], axis=1)
	data_test = pd.read_csv('data/test.csv', sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

	print('Applying to train...')
	# Length of questions
	data_train['len_q1'] = data_train.question1.apply(lambda x: len(str(x)))
	data_train['len_q2'] = data_train.question2.apply(lambda x: len(str(x)))
	data_train['diff_len'] = data_train.len_q1 - data_train.len_q2
	data_train['len_char_q1'] = data_train.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data_train['len_char_q2'] = data_train.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data_train['len_word_q1'] = data_train.question1.apply(lambda x: len(str(x).split()))
	data_train['len_word_q2'] = data_train.question2.apply(lambda x: len(str(x).split()))
	data_train['common_words'] = data_train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

	# Fuzz package to compute some ratio of string similarity between question 1 et question 2.
	data_train['fuzz_qratio'] = data_train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_WRatio'] = data_train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_partial_ratio'] = data_train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_partial_token_set_ratio'] = data_train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_partial_token_sort_ratio'] = data_train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_token_set_ratio'] = data_train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_train['fuzz_token_sort_ratio'] = data_train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

	# Embedding of the questions
	model=word_embedding_model_glove
	norm_model = word_embedding_model_glove
	norm_model.init_sims(replace=True)

	data_train['wmd'] = data_train.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

	question1_vectors = np.zeros((data_train.shape[0], 300))
	error_count = 0

	for i, q in tqdm(enumerate(data_train.question1.values)):
	    question1_vectors[i, :] = sent2vec(q)

	question2_vectors  = np.zeros((data_train.shape[0], 300))
	for i, q in tqdm(enumerate(data_train.question2.values)):
	    question2_vectors[i, :] = sent2vec(q)

	# Word embedding of the questions and compute different distances: WMD, cosine
	data_train['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_train['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
	data_train['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
	data_train['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
	data_train['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

	pickle.dump(question1_vectors, open('data/q1_glove.pkl', 'wb'), -1)
	pickle.dump(question2_vectors, open('data/q2_glove.pkl', 'wb'), -1)

	print('Writing train features...')
	data_train.to_csv('data/train_features_glove.csv', index=False)



	print('Applying to test...')
	# Length of questions
	data_test['len_q1'] = data_test.question1.apply(lambda x: len(str(x)))
	data_test['len_q2'] = data_test.question2.apply(lambda x: len(str(x)))
	data_test['diff_len'] = data_test.len_q1 - data_test.len_q2
	data_test['len_char_q1'] = data_test.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data_test['len_char_q2'] = data_test.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
	data_test['len_word_q1'] = data_test.question1.apply(lambda x: len(str(x).split()))
	data_test['len_word_q2'] = data_test.question2.apply(lambda x: len(str(x).split()))
	data_test['common_words'] = data_test.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

	# Fuzz package to compute some ratio of string similarity between question 1 et question 2.
	data_test['fuzz_qratio'] = data_test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_WRatio'] = data_test.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_partial_ratio'] = data_test.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_partial_token_set_ratio'] = data_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_partial_token_sort_ratio'] = data_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_token_set_ratio'] = data_test.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
	data_test['fuzz_token_sort_ratio'] = data_test.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

	# Embedding of the questions
	norm_model = model
	norm_model.init_sims(replace=True)
	data_test['wmd'] = data_test.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

	question1_vectors = np.zeros((data_test.shape[0], 300))
	error_count = 0

	for i, q in tqdm(enumerate(data_test.question1.values)):
	    question1_vectors[i, :] = sent2vec(q)

	question2_vectors  = np.zeros((data_test.shape[0], 300))
	for i, q in tqdm(enumerate(data_test.question2.values)):
	    question2_vectors[i, :] = sent2vec(q)

	# Word embedding of the questions and compute different distances: WMD, cosine
	data_test['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
	data_test['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
	data_test['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
	data_test['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
	data_test['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

	pickle.dump(question1_vectors, open('data/q1_glove_test.pkl', 'wb'), -1)
	pickle.dump(question2_vectors, open('data/q2_glove_test.pkl', 'wb'), -1)

	print('Writing test features...')
	data_test.to_csv('data/test_features_glove.csv', index=False)

