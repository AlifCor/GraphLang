import numpy as np
import itertools

from sklearn.metrics import f1_score


def find_best_perm(y_true, y_pred, metric=lambda y1, y2: f1_score(y1, y2, average='weighted')):
    scores = []
    permutations = list(itertools.permutations(range(len(set(y_true)))))
    for perm in permutations:
        apply_perm = np.vectorize(lambda x: perm[x])
        score = metric(y_true, apply_perm(y_pred))
        scores.append(score)

    return permutations[np.argmax(scores)]
