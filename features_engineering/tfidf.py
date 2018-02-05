import functools
from collections import defaultdict
from collections import Counter
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import os


def word_match_share(row, stops=None):
    """
    This function compute the number of the shared words between the question1 and question2 excluding stop word, 
    it is normalized by the total length of the question 1 and 2.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.
        stops: list of optional words to avoid.

    Return:

    """
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

'''
This function defines the jaccard index as the size of the intersection of q1 and q2 divided by the size of 
the union:
'''
def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

'''
This function computes the number of common words between q1 and q2
'''
    
def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))
'''
This function computes the number of unique words of q1 and q2 combined
'''    
def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

'''
This function computes the number of unique words of q1 and q2 combined excluding stop words
'''  
def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

'''
This function computes the difference of length between q1 and q2
''' 
def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

'''
This function computes the ration of length between q1 and q2
''' 
def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

'''
This function computes the absolute difference of length between q1 and q2
''' 
def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2
        
'''
This function computes the difference of length between q1 and q2 excluding stop words
'''
def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))
 
def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

'''
This function is a bolean that computes weither q1 and q2 have the same start or not
'''
def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

'''
This function returns the difference of length between the characters of q1 and q2
'''
def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

'''
This function returns the ratio of length between the characters of q1 and q2
'''
def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))

'''
This function returns weights to attribute to words with high frequency
'''

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
'''
This function returns the ratio of weights of shared words between q1 and q2 and the total weights (it is based
on attributing weights according to frequencies which explains the name tfidf). Here we exclude stop words
'''  
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
'''
    This function has the same idea as the previous function but here we include stop words
'''
def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

'''
This function allows to organize the different features in a pandas table in order to add theme to the data
'''
def build_features(data, stops, weights):
    X = pd.DataFrame()

    print('world_match')
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True) #1

    print('tfidf')
    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    print('tfidf_wm_stops')
    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    print("jaccard, wc_diff; wc_ratio, wc_diff_unique, wc_ratio_unique")
    X['jaccard'] = data.apply(jaccard, axis=1, raw=True) #4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    print("wc_diff_unq_stop, wc_ratio_unique_stop")
    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    print('same_start, char_diff')
    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    print('char_diff_unq_stop')
    f = functools.partial(char_diff_unique_stop, stops=stops) 
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

    print('total_unique_words')
#     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    print('total_unq_words_stop')
    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16
    
    print('char_ratio')
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17    

    return X


def generate_tfidf(path): 
    df_train =pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    df_train = df_train.fillna(' ')

    df_test=  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])
    ques = pd.concat([df_train[['question1', 'question2']], \
        df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
            q_dict[ques.question1[i]].add(ques.question2[i])
            q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return(len(q_dict[row['question1']]))

    def q2_freq(row):
        return(len(q_dict[row['question2']]))

    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
    df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
    df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

    df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
    df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
    df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

    test_leaky = df_test.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]
    del df_test

    train_leaky = df_train.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]

    # explore
    stops = set(stopwords.words("english"))

    df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
    df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    print('Building Features :')
    x_train = build_features(df_train, stops, weights)
    x_train = pd.concat((x_train, train_leaky), axis=1)

    df_test =pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])
    df_test = df_test.fillna(' ')

    df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
    df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

    x_test = build_features(df_test, stops, weights)
    x_test = pd.concat((x_test, test_leaky), axis=1)


    print('Writing train features...')
    x_train.to_csv(os.path.join(path,'train_tfidf.csv'))
    print('Writing test features...')    
    x_test.to_csv(os.path.join(path,'test_tfidf.csv'))
    print('CSV written ! see: ', path, " | suffix: ", "_tfidf.csv")