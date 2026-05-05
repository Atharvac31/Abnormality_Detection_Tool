import numpy as np
class SimpleCAGE:

    def __init__(self, lf_acc_prior=0.75):
        self.lf_acc_prior = lf_acc_prior


    def _compute_scores(self, L, C):

        n_samples, n_lfs = L.shape
        scores = np.zeros(n_samples)

        base_weight = np.log(self.lf_acc_prior / (1 - self.lf_acc_prior))

        for i in range(n_samples):

            score = 0

            for j in range(n_lfs):

                label = L[i, j]
                conf = C[i, j]

                if label == -1:
                    continue

                weight = base_weight * conf

                if label == 1:
                    score += weight
                else:
                    score -= weight

            scores[i] = score

        return scores


    def predict_proba(self, L, C):

        scores = self._compute_scores(L, C)
        probs = 1 / (1 + np.exp(-scores))

        return np.vstack([1 - probs, probs]).T


    def predict(self, L, C):

        probs = self.predict_proba(L, C)
        return np.argmax(probs, axis=1)