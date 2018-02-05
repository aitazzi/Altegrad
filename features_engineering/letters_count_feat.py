import string
import pandas as pd
import os

def generate_letters_count_features(path):
    """
    This function count the number of letters (from 'a' to 'z') and the number of voyelsin q1 and q2, it counts also 
    the number of spaces and words for Quora questions data.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.

    Return:

    """
    #word features
    train =pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    
    train['num_space_q1'] = train.question1.apply(lambda x: str(x).count(' '))
    train['num_space_q2'] = train.question2.apply(lambda x: str(x).count(' '))
    train['num_word_q1'] = train.question1.apply(lambda x: len(str(x).split()))
    train['num_word_q2'] = train.question2.apply(lambda x: len(str(x).split()))
    #generate letter count
    for c in list(set(string.ascii_lowercase)):
        train['num_' + c + '_q1'] = train.question1.apply(lambda x: str(x).lower().count(c))
        train['num_' + c + '_q2'] = train.question2.apply(lambda x: str(x).lower().count(c))
    
    #ount the number of voyels  
    train['num_vowels_q1'] = train['num_a_q1'] + train['num_e_q1'] + train['num_i_q1'] + train['num_o_q1'] + train['num_u_q1']
    train['num_vowels_q2'] = train['num_a_q2'] + train['num_e_q2'] + train['num_i_q2'] + train['num_o_q2'] + train['num_u_q2']
    
    print('Writing train features...')	    
    
    train = train.drop(['qid1','qid2','id','question1','question2','is_duplicate'],axis=1)
    train.to_csv(os.path.join(path,'train_letters_count_feat.csv'))
    print('CSV written ! see: ', path, " | suffix: ", "_count_feat.csv")
    
    #test features
    test =pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])
    
    test['num_space_q1'] = test.question1.apply(lambda x: str(x).count(' '))
    test['num_space_q2'] = test.question2.apply(lambda x: str(x).count(' '))
    test['num_word_q1'] = test.question1.apply(lambda x: len(str(x).split()))
    test['num_word_q2'] = test.question2.apply(lambda x: len(str(x).split()))
    #generate letter count
    for c in list(set(string.ascii_lowercase)):
        test['num_' + c + '_q1'] = test.question1.apply(lambda x: str(x).lower().count(c))
        test['num_' + c + '_q2'] = test.question2.apply(lambda x: str(x).lower().count(c))
    
    #count the number of voyels  
    test['num_vowels_q1'] = test['num_a_q1'] + test['num_e_q1'] + test['num_i_q1'] + test['num_o_q1'] + test['num_u_q1']
    test['num_vowels_q2'] = test['num_a_q2'] + test['num_e_q2'] + test['num_i_q2'] + test['num_o_q2'] + test['num_u_q2']

    print('Writing test features...')	    
    
    test = test.drop(['qid1','qid2','id','question1','question2'],axis=1)
    test.to_csv(os.path.join(path,'test_letters_count_feat.csv'))
    print('CSV written ! see: ', path, " | suffix: ", "_count_feat.csv")