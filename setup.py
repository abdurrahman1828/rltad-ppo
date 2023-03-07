import setuptools

setuptools.setup(
    name="rltad-ppo",
    version='1.0.0',
    author="Abdur Rahman",
    author_email="ar2806@msstate.edu",
    description="Code for Learning from Fully-labeled to Unlabeled Multivariate Time Series: Proximal Policy Optimization-based Reinforcement Learning for Anomaly Detection",
    url="https://github.com/abdurrahman1828/rltad-ppo",
    keywords=["Anomaly detection", "time series", "multivariate", "interpretability", "Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=[
        'stable-baselines3[extra]',
    ],
    requires_python='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
)