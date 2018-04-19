import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

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
