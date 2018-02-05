import pandas as pd
import hashlib
import gc 
import os


def generate_pagerank(path):
    """
    Pagerank ranks the nodes in a graph structure, based on the linkages between them. 
    An edge shared with an important node makes it important as well,
    In this context, every node is a question in our dataset and an edge represents a question pair.

    Args:
        path: folder containing train.csv and test.csv and to write csv features file.

    Return:

    """

    df_train = pd.read_csv(os.path.join(path,'train.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2","is_duplicate"])
    df_test = pd.read_csv(os.path.join(path,'test.csv'), sep=',',names = ["id", "qid1", "qid2", "question1","question2"])

    # Generating a graph of Questions and their neighbors (computes a dictionary of questions, where each 
    # key-value pair is a question and its neighboring questions(in a list).)
    def generate_qid_graph_table(row):
        # Using hashing algorithm MD5 which gives the authenticating digital signatures for each question
        hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
        hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
        
        qid_graph.setdefault(hash_key1, []).append(hash_key2)
        qid_graph.setdefault(hash_key2, []).append(hash_key1)
        return

    qid_graph = {}
    print('Apply to train...')
    df_train.apply(generate_qid_graph_table, axis=1)

    print('Apply to test...')
    df_test.apply(generate_qid_graph_table, axis=1)
    
    # Applying the page rank formula to update the pagerank value of each node.
    # pagerank of a node is defined as the sum of a certain ratio of all its neighbors 
    # This function defines then the sequence of pageRank values that are updated at each iteration until the convergence
    def pagerank():
        MAX_ITER = 20
        d = 0.85
        # Initializing -- every node gets a uniform value!
        pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
        num_nodes = len(pagerank_dict)
        for iter in range(0, MAX_ITER):
            for node in qid_graph:
                local_pr = 0
                for neighbor in qid_graph[node]:
                    local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])
                pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr
        return pagerank_dict

    print('Main PR generator...')
    pagerank_dict = pagerank()
    
    # Finally we compute the pageRank features 
    def get_pagerank_value(row):
        q1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
        q2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
        s = pd.Series({
            "q1_pr": pagerank_dict[q1],
            "q2_pr": pagerank_dict[q2]
        })
        return s



    print('Apply to train...')
    pagerank_feats_train = df_train.apply(get_pagerank_value, axis=1)

    print('Writing train features...')
    pagerank_feats_train.to_csv(os.path.join(path,"train_pagerank.csv"), index=False)
    del df_train
    gc.collect()

    print('Apply to test...')
    pagerank_feats_test = df_test.apply(get_pagerank_value, axis=1)

    print('Writing test features...')
    pagerank_feats_test.to_csv(os.path.join(path,"test_pagerank.csv"), index=False)

    print('CSV written ! see: ', path, " | suffix: ", "_pagerank.csv")


