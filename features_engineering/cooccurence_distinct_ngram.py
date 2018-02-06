import pandas as pd
import numpy as np
import os
from nltk import ngrams
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from ngram import NGram
from tqdm import tqdm

def generate_cooccurence_distinct_ngram(path, n=2):
    """
    Generate n-gram features (with and without stopwords removed) for question pairs data. 
    Features will be written in a csv file in path folder.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.
        n: number of word for the ngram.

    Return:
        
    """

    # Load training and test set
    train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

    # Drop useless columns
    train = train.drop(['id','qid1','qid2','is_duplicate'], axis=1)
    test = test.drop(['id','qid1','qid2'], axis=1)

    # Set up the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    print('Applying to train...')
    for index,row in tqdm(train.iterrows()):
        
        question1 = train['question1'][index]
        question2 = train['question2'][index]
        
        # Tokenize each question
        tokenize1 = tokenizer.tokenize(question1)
        tokenize2 = tokenizer.tokenize(question2)

        # Create n-grams
        ngram1 = [gram for gram in ngrams(tokenize1, n)]
        ngram2 = [gram for gram in ngrams(tokenize2, n)]

        # Remove stopwords
        tokenize_no_stopword1 = [w for w in tokenize1 if not w in stop_words]
        tokenize_no_stopword2 = [w for w in tokenize2 if not w in stop_words]

        # Create n-grams without stopwords
        ngram_no_stopword1 = [gram for gram in ngrams(tokenize_no_stopword1, n)]
        ngram_no_stopword2 = [gram for gram in ngrams(tokenize_no_stopword2, n)]

        # Count cooccurence and distincts n-grams without stopwords
        cooccurence_no_stopword = 0
        distinct_no_stopword = 0
        for gram1 in ngram_no_stopword1:
            n1 = NGram(gram1)
            for gram2 in ngram_no_stopword2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct_no_stopword += 1
                elif len(inter) == 2:
                    cooccurence_no_stopword += 1
                    
        train.loc[index,str(n)+'gram_nostpwrd_cooccurence'] = cooccurence_no_stopword
        train.loc[index,str(n)+'gram_nostpwrd_distinct'] = distinct_no_stopword

        # Count cooccurence and distincts n-grams with stopwords        
        cooccurence = 0
        distinct = 0
        for gram1 in ngram1:
            n1 = NGram(gram1)
            for gram2 in ngram2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct += 1
                elif len(inter) == 2:
                    cooccurence += 1

        train.loc[index,str(n)+'gram_cooccurence'] = cooccurence
        train.loc[index,str(n)+'gram_distinct'] = distinct

    # Drop useless columns to save in csv
    train = train.drop(['question1', 'question2'], axis=1)
    print('Writing train features...')
    train.to_csv(os.path.join(path,'train_'+str(n)+'gram_feat.csv'))


    print('Applying to test...')
    for index,row in tqdm(test.iterrows()):
        
        question1 = test['question1'][index]
        question2 = test['question2'][index]

        # Tokenize each question
        tokenize1 = tokenizer.tokenize(question1)
        tokenize2 = tokenizer.tokenize(question2)

        # Create n-grams
        ngram1 = [gram for gram in ngrams(tokenize1, n)]
        ngram2 = [gram for gram in ngrams(tokenize2, n)]

        # Remove stopwords
        tokenize_no_stopword1 = [w for w in tokenize1 if not w in stop_words]
        tokenize_no_stopword2 = [w for w in tokenize2 if not w in stop_words]

        # Create n-grams without stopwords
        ngram_no_stopword1 = [gram for gram in ngrams(tokenize_no_stopword1, n)]
        ngram_no_stopword2 = [gram for gram in ngrams(tokenize_no_stopword2, n)]

        # Count cooccurence and distincts n-grams without stopwords
        cooccurence_no_stopword = 0
        distinct_no_stopword = 0
        for gram1 in ngram_no_stopword1:
            n1 = NGram(gram1)
            for gram2 in ngram_no_stopword2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct_no_stopword += 1
                elif len(inter) == 2:
                    cooccurence_no_stopword += 1
                    
        test.loc[index,str(n)+'gram_nostpwrd_cooccurence'] = cooccurence_no_stopword
        test.loc[index,str(n)+'gram_nostpwrd_distinct'] = distinct_no_stopword

        # Count cooccurence and distincts n-grams with stopwords
        cooccurence = 0
        distinct = 0
        for gram1 in ngram1:
            n1 = NGram(gram1)
            for gram2 in ngram2:
                n2 = NGram(gram2)
                inter = n1.intersection(n2)
                if len(inter) == 0:
                    distinct += 1
                elif len(inter) == 2:
                    cooccurence += 1

        test.loc[index,str(n)+'gram_cooccurence'] = cooccurence
        test.loc[index,str(n)+'gram_distinct'] = distinct

    test = test.drop(['question1', 'question2'], axis=1)
    print('Writing test features...')
    test.to_csv(os.path.join(path,'test_'+str(n)+'gram_feat.csv'))

    print('CSV written ! see: ', path, " | suffix: ", "_"+str(n)+"gram_feat.csv")




