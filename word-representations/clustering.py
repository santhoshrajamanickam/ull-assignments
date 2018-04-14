import numpy as np
from sklearn.cluster import KMeans

def cluster_words(embedding_dict, frequent_nouns, num_clusters):

    labels = dict()

    for emb in embedding_dict:
        print(emb)

        embeddings = []

        for word in frequent_nouns:
            word_idx = embedding_dict[emb].word2idx[word]
            word_emb = embedding_dict[emb].vectors[word_idx, :]
            embeddings.append(word_emb)

        X = np.array(embeddings)

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels[emb] = kmeans.labels_

    return labels['BOW2'], labels['BOW5'], labels['Dependency']
