import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_static_score(new_embedding, stored_embeddings):

    if not stored_embeddings:
        return 0.0
    for e in stored_embeddings:
        print(e)
    print("new embedding", new_embedding)
    sims = []

    for emb in stored_embeddings:
        sim = cosine_similarity(
            [new_embedding],
            [emb]
        )[0][0]

        sims.append(sim)

    # User level similarity
    S_static_user = np.mean(sims)

    return float(S_static_user)
