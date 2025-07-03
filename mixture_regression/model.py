from scipy.stats import norm
import numpy as np


class MixtureOfLinearRegressions:
    def __init__(self, n_components, n_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = np.random.default_rng(random_state)
        self.weights_ = None
        self.betas_ = None
        self.sigmas_ = None
        self.resp_ = None
        self.log_likelihood_ = []

    def _initialize_parameters(self, X, y):
        n, p = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.betas_ = [self.random_state.normal(0, 1, size=p) for _ in range(self.n_components)]
        self.sigmas_ = np.full(self.n_components, np.std(y))

    def _e_step(self, X, y):
        n = X.shape[0]
        resp = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            mu = X @ self.betas_[k]
            resp[:, k] = self.weights_[k] * norm.pdf(y, loc=mu, scale=self.sigmas_[k])
        resp /= resp.sum(axis=1, keepdims=True)
        self.resp_ = resp

    def _m_step(self, X, y):
        n, p = X.shape
        for k in range(self.n_components):
            r_k = self.resp_[:, k]
            R = np.diag(r_k)
            X_weighted = X.T @ R @ X
            y_weighted = X.T @ R @ y
            beta_k = np.linalg.solve(X_weighted, y_weighted)
            self.betas_[k] = beta_k
            y_pred = X @ beta_k
            sigma_k = np.sqrt((r_k * (y - y_pred) ** 2).sum() / r_k.sum())
            self.sigmas_[k] = sigma_k
            self.weights_[k] = r_k.mean()

    def _compute_log_likelihood(self, X, y):
        likelihood = 0
        for k in range(self.n_components):
            mu = X @ self.betas_[k]
            likelihood += self.weights_[k] * norm.pdf(y, loc=mu, scale=self.sigmas_[k])
        return np.sum(np.log(likelihood))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._initialize_parameters(X, y)

        for iteration in range(self.n_iter):
            self._e_step(X, y)
            self._m_step(X, y)
            log_likelihood = self._compute_log_likelihood(X, y)
            self.log_likelihood_.append(log_likelihood)
            if iteration > 0 and np.abs(self.log_likelihood_[-1] - self.log_likelihood_[-2]) < self.tol:
                break
        return self

    def predict_cluster(self):
        return np.argmax(self.resp_, axis=1)

    def bic(self, X, y):
        n, p = X.shape
        n_params = self.n_components * (p + 2) - 1  # each component has p betas + 1 sigma + 1 weight (minus 1 for sum to 1)
        log_likelihood = self._compute_log_likelihood(X, y)
        return n_params * np.log(n) - 2 * log_likelihood

    def get_log_likelihood(self):
        return self.log_likelihood_[-1]

    def get_pvalues(self, X, y):
        pvals = []
        for k in range(self.n_components):
            r_k = self.resp_[:, k]
            X_weighted = X * np.sqrt(r_k[:, None])
            y_weighted = y * np.sqrt(r_k)
            beta_k = self.betas_[k]
            sigma_k = self.sigmas_[k]
            XtX_inv = np.linalg.inv(X_weighted.T @ X_weighted)
            se = np.sqrt(np.diag(XtX_inv)) * sigma_k
            t_stats = beta_k / se
            p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
            pvals.append(p_values)
        return np.array(pvals)
