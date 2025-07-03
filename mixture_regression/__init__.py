# mixture_regression/__init__.py
from .model import MixtureOfLinearRegressions
from .bic_selector import MinBIC
from .runner import MixtureModelRunner
from .generate import MixtureRegressionDataGenerator  # ← updated import