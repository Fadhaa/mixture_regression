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

📘 Full Application Example: Mixture of Linear Regressions with BIC Selection
🧮 Step 1: Import Modules
We begin by importing the necessary modules from the mixture_regression package:

###python

from mixture_regression import generate, model
from mixture_regression import bic_selector, runner
🎯 Step 2: Define Custom Regression Coefficients
Each cluster (component) will have its own set of regression coefficients (betas), including an intercept term. Here we define three clusters with known coefficients:

###python

custom_betas = [
    [0.5, 1.5, 2, 3],   # Cluster 0
    [1, 2, 3, 4],       # Cluster 1
    [2, 3, 4, 6]        # Cluster 2
]
🧪 Step 3: Generate Simulated Data
We generate 1,000 observations with 3 features and 3 clusters using the MixtureRegressionDataGenerator:

###python

generator = generate.MixtureRegressionDataGenerator(
    n_samples=1000,
    n_features=3,
    n_clusters=3,
    cluster_probs=[0.4, 0.3, 0.3],  # Probability of each cluster
    noise_std=1.0,
    random_state=42,
    betas=custom_betas
)

df = generator.generate()
print(generator.get_true_betas())
Output:

[[0.5, 1.5, 2, 3], [1, 2, 3, 4], [2, 3, 4, 6]]
✅ This confirms that the true underlying structure has been used to simulate the data.

🧠 Step 4: Fit the Model and Select Best Number of Components
Now we fit the mixture model and let the algorithm automatically select the best number of components using the Bayesian Information Criterion (BIC):

###python

runner = runner.MixtureModelRunner(df)
runner.run()
🖨️ Output: Model Fitting and BIC Scores


Components: 1, BIC: 4138.50
Components: 2, BIC: 3774.48
Components: 3, BIC: 3731.61
Components: 4, BIC: 3752.48
Components: 5, BIC: 3771.93

✅ Best number of components based on BIC: 3

📊 Final Model (k = 3)
BIC: 3770.84

Component 1:
  Weight (π):     0.2815
  Coefficients (β): [1.9249101  3.10512153 3.91060425 6.04186013]
  Std Dev (σ):     0.9201
----------------------------------------
Component 2:
  Weight (π):     0.4283
  Coefficients (β): [0.42579598 1.46916441 1.98903705 3.1043965 ]
  Std Dev (σ):     0.9366
----------------------------------------
Component 3:
  Weight (π):     0.2901
  Coefficients (β): [1.19721701 1.99912275 3.15040332 3.97245209]
  Std Dev (σ):     0.9920
🔍 Interpretation
✅ The algorithm correctly identifies 3 components as the optimal number using BIC.

📉 BIC decreases until 3 components, then increases, indicating the optimal model complexity.

🧮 The estimated coefficients (β) are close to the true betas you defined:

[0.5, 1.5, 2, 3]

[1, 2, 3, 4]

[2, 3, 4, 6]

🧪 The weights (π) indicate the mixture proportions (close to your original [0.4, 0.3, 0.3]).

📐 The standard deviations (σ) reflect noise in each sub-model.
