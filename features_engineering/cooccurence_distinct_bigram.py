import pandas as pd
import numpy as np
import os
from nltk import ngrams
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from ngram import NGram
from tqdm import tqdm

def generate_cooccurence_distinct_bigram(path):
    """
    Generate bigram features for Quora question data. Features will be written in a csv file in path folder.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.

    Return:
        
    """
    train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

    train = train.drop(['id','qid1','qid2','is_duplicate'], axis=1)
    test = test.drop(['id','qid1','qid2'], axis=1)

    train['bigram_coocurence'] = np.nan; train['bigram_distinct'] = np.nan
    train['bigram_nostpwrd_coocurence'] = np.nan; train['bigram_nostpwrd_coocurence'] = np.nan

    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    print('Applying to train...')
    for index,row in tqdm(train.iterrows()):
        
        question1 = train['question1'][index]
        question2 = train['question2'][index]
        
        tokenize1 = tokenizer.tokenize(question1)
        tokenize2 = tokenizer.tokenize(question2)
        bigram1 = [gram for gram in ngrams(tokenize1, 2)]
        bigram2 = [gram for gram in ngrams(tokenize2, 2)]
        tokenize_no_stopword1 = [w for w in tokenize1 if not w in stop_words]
        tokenize_no_stopword2 = [w for w in tokenize2 if not w in stop_words]
        bigram_no_stopword1 = [gram for gram in ngrams(tokenize_no_stopword1, 2)]
        bigram_no_stopword2 = [gram for gram in ngrams(tokenize_no_stopword2, 2)]

        cooccurence_no_stopword = 0
        distinct_no_stopword = 0
        for gram1 in bigram_no_stopword1:
            n1 = NGram(gram1)
            for gram2 in bigram_no_stopword2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct_no_stopword += 1
                elif len(inter) == 2:
                    cooccurence_no_stopword += 1
                    
        train.loc[index,'bigram_nostpwrd_coocurence'] = cooccurence_no_stopword
        train.loc[index,'bigram_nostpwrd_distinct'] = distinct_no_stopword
        
        cooccurence = 0
        distinct = 0
        for gram1 in bigram1:
            n1 = NGram(gram1)
            for gram2 in bigram2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct += 1
                elif len(inter) == 2:
                    cooccurence += 1

        train.loc[index,'bigram_coocurence'] = cooccurence
        train.loc[index,'bigram_distinct'] = distinct

    train = train.drop(['question1', 'question2'], axis=1)
    print('Writing train features...')
    train.to_csv(os.path.join(path,'train_bigram_feat.csv'))


    test['bigram_coocurence'] = np.nan; test['bigram_distinct'] = np.nan
    test['bigram_nostpwrd_coocurence'] = np.nan; test['bigram_nostpwrd_coocurence'] = np.nan

    print('Applying to test...')
    for index,row in tqdm(test.iterrows()):
        
        question1 = test['question1'][index]
        question2 = test['question2'][index]
        
        tokenize1 = tokenizer.tokenize(question1)
        tokenize2 = tokenizer.tokenize(question2)
        bigram1 = [gram for gram in ngrams(tokenize1, 2)]
        bigram2 = [gram for gram in ngrams(tokenize2, 2)]
        tokenize_no_stopword1 = [w for w in tokenize1 if not w in stop_words]
        tokenize_no_stopword2 = [w for w in tokenize2 if not w in stop_words]
        bigram_no_stopword1 = [gram for gram in ngrams(tokenize_no_stopword1, 2)]
        bigram_no_stopword2 = [gram for gram in ngrams(tokenize_no_stopword2, 2)]

        cooccurence_no_stopword = 0
        distinct_no_stopword = 0
        for gram1 in bigram_no_stopword1:
            n1 = NGram(gram1)
            for gram2 in bigram_no_stopword2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct_no_stopword += 1
                elif len(inter) == 2:
                    cooccurence_no_stopword += 1
                    
        test.loc[index,'bigram_nostpwrd_coocurence'] = cooccurence_no_stopword
        test.loc[index,'bigram_nostpwrd_distinct'] = distinct_no_stopword
        
        cooccurence = 0
        distinct = 0
        for gram1 in bigram1:
            n1 = NGram(gram1)
            for gram2 in bigram2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct += 1
                elif len(inter) == 2:
                    cooccurence += 1

        test.loc[index,'bigram_coocurence'] = cooccurence
        test.loc[index,'bigram_distinct'] = distinct

    test = test.drop(['question1', 'question2'], axis=1)
    print('Writing test features...')
    test.to_csv(os.path.join(path,'test_bigram_feat.csv'))

    print('CSV written ! see: ', path, " | suffix: ", "_bigram_feat.csv")




