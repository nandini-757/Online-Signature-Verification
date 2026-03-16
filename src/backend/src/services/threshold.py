import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.dynamic_matcher import compute_dynamic_score


def compute_threshold(
    static_embeddings,
    dynamic_features,
    alpha=0.7
):

    pair_scores = []

    for i in range(len(static_embeddings)):
        for j in range(i + 1, len(static_embeddings)):

            # ===============================
            # STATIC SIMILARITY
            # ===============================
            S_static = cosine_similarity(
                [static_embeddings[i]],
                [static_embeddings[j]]
            )[0][0]

            # ===============================
            # DYNAMIC HYBRID SCORE
            # ===============================
            Dynamic_Score = compute_dynamic_score(
                dynamic_features[i],
                [dynamic_features[j]]
            )

            # ===============================
            # FINAL FUSION SCORE
            # ===============================
            Final = (
                alpha * S_static
                +
                (1 - alpha) * Dynamic_Score
            )

            pair_scores.append(Final)

    pair_scores = np.array(pair_scores)

    threshold = (
        np.mean(pair_scores)
        -
        0.5 * np.std(pair_scores)
    )

    return float(threshold)
