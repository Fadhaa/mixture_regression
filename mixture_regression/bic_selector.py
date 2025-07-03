
import numpy as np
import pandas as pd
from .model import MixtureOfLinearRegressions




class MinBIC:
    def __init__(self, df, response_col='y', random_state=42):
        self.df = df
        self.response_col = response_col
        self.random_state = random_state
        self.bic_scores = []
        self.k_values = []
        self.models = []

    def _prepare_data(self):
        # Automatically determine feature columns
        self.feature_cols = [col for col in self.df.columns if col != self.response_col]
        X_raw = self.df[self.feature_cols].values
        y = self.df[self.response_col].values
        # Add intercept
        X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])
        return X, y

    def fit(self):
        X, y = self._prepare_data()

        k = 1
        min_bic = np.inf
        min_k = None
        bic_window = []

        # Loop until we pass the minimum by 2 more k's
        while True:
            model = MixtureOfLinearRegressions(n_components=k, random_state=self.random_state)
            model.fit(X, y)
            bic = model.bic(X, y)

            # Store results
            self.bic_scores.append(bic)
            self.k_values.append(k)
            self.models.append(model)

            # Print progress
            print(f"Components: {k}, BIC: {bic:.2f}")

            # Track minimum BIC
            if bic < min_bic:
                min_bic = bic
                min_k = k
                bic_window = [k]
            elif k - min_k >= 2:
                break
            else:
                bic_window.append(k)

            k += 1

        self.best_k = min_k
        return self

    def get_bic_scores(self):
        return pd.DataFrame({
            'Components': self.k_values,
            'BIC': self.bic_scores
        })

    def get_best_model(self):
        return self.models[self.k_values.index(self.best_k)]

