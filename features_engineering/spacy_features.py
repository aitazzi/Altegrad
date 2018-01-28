import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def generate_spacy_features(path):
    train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    test =  pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

    train = train.drop(['id','qid1','qid2','is_duplicate'], axis=1)
    test = test.drop(['id','qid1','qid2'], axis=1)

    train['spacy_similarity'] = np.nan

    print('Applying to train...')
    for index,row in tqdm(train.iterrows()):
        question1 = train['question1'][index]
        question2 = train['question2'][index]

        question1_nlp = nlp(question1)
        question2_nlp = nlp(question2)

        train.loc[index,'spacy_similarity'] = question1_nlp.similarity(question2_nlp)

    train = train.drop(['question1', 'question2'], axis=1)
    print('Writing train features...')
    train.to_csv(os.path.join(path,'train_spacy_features.csv'))


    test['spacy_similarity'] = np.nan

    print('Applying to test...')
    for index,row in tqdm(test.iterrows()):
        question1 = test['question1'][index]
        question2 = test['question2'][index]

        question1_nlp = nlp(question1)
        question2_nlp = nlp(question2)

        test.loc[index,'spacy_similarity'] = question1_nlp.similarity(question2_nlp)

    test = test.drop(['question1', 'question2'], axis=1)
    print('Writing test features...')
    test.to_csv(os.path.join(path,'test_spacy_features.csv'))

    print('CSV written ! see: ', path, " | suffix: ", "_spacy_features.csv")