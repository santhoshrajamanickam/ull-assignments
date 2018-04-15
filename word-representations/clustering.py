import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ggplot import *
import pandas as pd

def cluster_words(embedding_dict, frequent_nouns, num_clusters):
    """
    Returns the cluster labels for the words clustered using each of the three word embedding models.
    Args:
        - embedding_dict (dict): embedding dictionary containing the three word embedding representation of the words
        - frequent_nouns (list): list containing the words that we have to cluster
        - num_clusters (int): the number of clusters
    Returns:
        embedding dictionary containing only the frequent_nouns, three lists containing
        the labels that resulted in the clustering using the three word embedding representation
    """

    labels = dict()
    embeddings = defaultdict(list)

    for emb in embedding_dict:
        # print(emb)

        for word in frequent_nouns:
            word_idx = embedding_dict[emb].word2idx[word]
            word_emb = embedding_dict[emb].vectors[word_idx, :]
            embeddings[emb].append(word_emb)

        X = np.array(embeddings[emb])

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels[emb] = kmeans.labels_

    return embeddings, labels['BOW2'], labels['BOW5'], labels['Dependency']


def print_cluster_words(cluster, words, labels):
    """
    Prints the 10 or less words with cluster as its label

    Args:
        - cluster (int): number denoting the cluster we get words from
        - words (list): list containing the words from which we have to select the word belonging to the cluster
        - labels (list): list containing each of the words labels
    """

    cluster_words = []
    num_words = 0

    for i, cluster_num in enumerate(labels):
        if num_words > 10:
            break

        if cluster_num == cluster:
            cluster_words.append(words[i])
            num_words = num_words + 1

    print(cluster_words)


def visualize_pca(embeddings, labels, emb_type):
    """
    Uses PCA to reduce the dimension of the features and then visualize using the
    labels to assign different colors to different cluster words. Uses random 200
    words for brevity and plots a 2D plot of the words.
    Args:
        - embeddings (list): embedding list containing the features for each of the words in the vocabulary
        - labels (list): list containing the labels of the words in the vocabulary
        - emb_type (str): specifies the type of word embedding representation
    """

    feat_cols = [ 'feature'+str(i) for i in range(len(embeddings[1])) ]

    df = pd.DataFrame(embeddings,columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]

    # print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))

    graph_title = emb_type + ": 1st and 2nd principal components colored by cluster belonging words"

    chart = ggplot( df.loc[rndperm[:200],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle(graph_title)
    print(chart)
