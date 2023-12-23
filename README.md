## A Proximal Policy Optimization-based Reinforcement Learning Model for Multivariate Time Series Anomaly Detection

### Abstract
Efficient anomaly detection in multivariate time series data is a challenging task due to (1) data labeling, and (2) intricate patterns and complex correlations among the data channels. The unsupervised deep learning approaches that leverage reconstruction error to detect anomalies struggle when anomalies are present in the training data. Moreover, only a few of them can effectively capture the historical pattern and correlation due to the dynamic nature of time series data. In this study, a Reinforcement Learning-based multivariate Time series Anomaly Detection (RLTAD) method has been proposed. RLTAD employs the proximal policy optimization (PPO) method to optimize the underlying policy to discourage large policy updates and avoid collapsing the policy to a suboptimal one. As a result, the RLTAD learns to generate policies that can capture the dynamic and complex correlations of multivariate time series.  By leveraging a pseudo-label generator, RLTAD is uniquely designed as a universal model so that it can work on fully labeled data, partially labeled data, and unlabeled data. Results demonstrate that the RLTAD achieved statistically significant improvement in anomaly detection compared to the baselines. For instance, RLTAD obtained a 3.29\%, 1.85\%, and 17.58\% average improvements in F1-score across three real datasets. In addition, the computational complexity of RLTAD has been analyzed and the interpretability has been studied by employing the Gradient Class Activation Map (Grad-CAM).



<img width="800" src="./imgs/rltad.png" alt="overview" />


## Installation
Make sure you have **python 3.8+** installed.
```
git clone https://github.com/abdurrahman1828/rltad-ppo.git
cd rltad-ppo
pip install -r requirments.txt
pip install -e .
```

## Scripts
Train and evaluate RLTAD with `main.py`. \
RL environment is defined in `env.py`. \
Required utility functions are in `utils.py`.

## Dataset sources
SKAB (Publicly available): [https://github.com/waico/SKAB](https://github.com/waico/SKAB) \
SWaT & WADI (Available on request): [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

