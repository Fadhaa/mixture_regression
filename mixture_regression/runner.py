import numpy as np
from .bic_selector import MinBIC
from .model import MixtureOfLinearRegressions
from .generate import MixtureRegressionDataGenerator

class MixtureModelRunner:
    def __init__(self, df, response_col='y', random_state=42):
        self.df = df
        self.response_col = response_col
        self.random_state = random_state

    def _prepare_data(self):
        self.feature_cols = [col for col in self.df.columns
                             if col not in [self.response_col, 'cluster']]
        X_raw = self.df[self.feature_cols].values
        y = self.df[self.response_col].values
        X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])
        return X, y
    def run(self):
        # Step 1: Use MinBIC to find best number of components
        bic_selector = MinBIC(self.df, response_col=self.response_col, random_state=self.random_state)
        bic_selector.fit()
        best_k = bic_selector.best_k
        print(f"\n✅ Best number of components based on BIC: {best_k}")

        # Step 2: Fit final model using best_k
        X, y = self._prepare_data()
        model = MixtureOfLinearRegressions(n_components=best_k, random_state=self.random_state)
        model.fit(X, y)
        bic_value = model.bic(X, y)

        # Step 3: Display all estimated parameters
        print(f"\n📊 Final Model (k = {best_k})")
        print(f"BIC: {bic_value:.2f}\n")

        for k in range(best_k):
            print(f"Component {k+1}:")
            print(f"  Weight (π):     {model.weights_[k]:.4f}")
            print(f"  Coefficients (β): {model.betas_[k]}")
            print(f"  Std Dev (σ):     {model.sigmas_[k]:.4f}")
            print("-" * 40)

        # Optionally return the final model object
        self.model = model
        self.best_k = best_k
        self.bic = bic_value
        return self


