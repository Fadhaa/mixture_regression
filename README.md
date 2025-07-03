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

