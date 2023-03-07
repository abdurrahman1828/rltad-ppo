import os
from stable_baselines3 import PPO


import pandas as pd

from utils import data_split
from env import Env

#parameter values
N = 4000000

#load data
data = pd.read_csv('D:/Abdur/Anomaly_Detection/alldata_skab.csv') #dataset csv
data = data.drop(['datetime','changepoint'], axis=1)
data = data.dropna()
data = data.reset_index(drop=True)

train, test = data_split(data, 0.5, 1)

train_env = Env(train)
test_env = Env(test)

log_path = os.path.join('Training', 'Log')
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=N)

obs = test_env.reset()
n_steps = 20000 #it should be more than number of windows in test set
rew = 0
act = []
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    act.append(action)
    obs, reward, done, info = test_env.step(action)
    rew+=reward
    if done:
        print("Goal reached!", "reward=", rew)
        print(act) # predictions
        break
test_env.close()


if __name__ == '__main__':
    print('Running...')


