from setuptools import setup, find_packages

setup(
    name='mixture_regression',
    version='0.1.0',
    description='EM-based mixture of linear regressions with BIC model selection',
    author='Your Name',
    author_email='your@email.com',
    url='https://github.com/yourusername/mixture_regression',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

