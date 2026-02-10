import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_threshold(embeddings):
    sims = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sims.append(cosine_similarity(
                [embeddings[i]], [embeddings[j]]
            )[0][0])
    return np.mean(sims) - np.std(sims)
354