import pandas as pd
import numpy as np

from scipy.stats import norm



##generate mixture data 



class MixtureRegressionDataGenerator:
    def __init__(self, n_samples=1000, n_features=3, n_clusters=3,
                 cluster_probs=None, noise_std=1.0, random_state=None, betas=None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.cluster_probs = cluster_probs if cluster_probs is not None else [1.0 / n_clusters] * n_clusters
        self.noise_std = noise_std
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        if betas is not None:
            assert len(betas) == n_clusters, "Length of betas must equal n_clusters"
            assert all(len(b) == n_features + 1 for b in betas), \
                f"Each beta must have length {n_features + 1} (intercept + features)"
            self.true_betas = betas
        else:
            self.true_betas = self._generate_true_betas()

    def _generate_true_betas(self):
        # Generate random coefficients for each cluster: (intercept + n_features)
        return [self._rng.normal(loc=0, scale=3, size=self.n_features + 1) for _ in range(self.n_clusters)]

    def generate(self):
        # Step 1: Assign cluster labels
        z = self._rng.choice(self.n_clusters, size=self.n_samples, p=self.cluster_probs)

        # Step 2: Generate independent variables
        X = self._rng.normal(0, 1, size=(self.n_samples, self.n_features))
        X_with_intercept = np.hstack([np.ones((self.n_samples, 1)), X])  # Add intercept term

        # Step 3: Generate dependent variable y
        y = np.zeros(self.n_samples)
        for k in range(self.n_clusters):
            idx = z == k
            y[idx] = X_with_intercept[idx] @ self.true_betas[k] + self._rng.normal(0, self.noise_std, size=idx.sum())

        # Step 4: Return as DataFrame
        df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(self.n_features)])
        df['y'] = y
        df['cluster'] = z
        return df

    def get_true_betas(self):
        return self.true_betas

