# 📊 Mixture Regression

A lightweight Python package for fitting **finite mixture models of linear regression** using the **Expectation-Maximization (EM)** algorithm — built entirely from scratch.

This is useful for modeling data with latent subpopulations, such as in clinical electronic health records (EHR), stratified outcomes, or hidden classes in regression problems.

---

## 🔧 Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/Fadhaa/mixture_regression.git

📦 Module Structure
generate: Generate synthetic regression mixture data

model: Core EM algorithm (MixtureOfLinearRegressions)

bic_selector: Automatically selects the best number of components using BIC

runner: Full pipeline to generate, fit, select, and print results

🚀 Quick Example
Step 1: Import modules
python
Copy
Edit
from mixture_regression import generate, model, bic_selector, runner
Step 2: Generate synthetic data with known true betas
python
Copy
Edit
custom_betas = [
    [0.5, 1.5, 2, 3],   # Cluster 1
    [1, 2, 3, 4],       # Cluster 2
    [2, 3, 4, 6]        # Cluster 3
]

generator = generate.MixtureRegressionDataGenerator(
    n_samples=1000,
    n_features=3,
    n_clusters=3,
    cluster_probs=[0.4, 0.3, 0.3],
    noise_std=1.0,
    random_state=42,
    betas=custom_betas
)

df = generator.generate()
print("True Betas:", generator.get_true_betas())
Step 3: Run the full EM model with BIC selection
python
Copy
Edit
runner = runner.MixtureModelRunner(df)
runner.run()
Example Output:
yaml
Copy
Edit
Components: 1, BIC: 4138.50
Components: 2, BIC: 3774.48
Components: 3, BIC: 3731.61
...

✅ Best number of components based on BIC: 3

📊 Final Model (k = 3)
BIC: 3731.61

Component 1:
  Weight (π):     0.2815
  Coefficients (β): [1.92 3.10 3.91 6.04]
  Std Dev (σ):     0.92
...
📈 Features
Mixture of linear regressions with EM (implemented from scratch)

Automatic selection of optimal number of components using BIC

Outputs weights, means, variances, and posterior responsibilities

P-values for variable significance per component (via Wald test)

Works well for simulated or real-world EHR data
