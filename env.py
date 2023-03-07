from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import random
import operator
from utils import read_data_as_matrix, scaling, sliding_wd
import numpy as np


class Env(Env):
    def __init__(self, data):
        # Actions space : 0 or 1
        self.action_space = Discrete(2)
        self.valid_label = {0, 1}
        self.number_of_channel = 8  # for skab dataset
        # Temperature array
        self.wnd_len = 30
        self.observation_space = Box(0, 1, shape=(self.wnd_len, self.number_of_channel))
        self.len = len(data.index)
        self.index = np.sort(random.sample(range(0, self.len), int(0.2 * self.len)))  # 0.2 means 20% unlabeled
        counter = 0
        for i in self.index:
            data.loc[data.index[i], 'anomaly'] = 500  # a number other than 0 and 1 to indicate absence of labels
            counter += 1
        print("Number of data point without labels: ", counter)
        X_train, labels, anomalies = read_data_as_matrix(data)

        self.X_train = X_train  # array
        self.labels = labels  # series, anomalies is also array
        self.scaled_data = scaling(data, data.columns[:-1])

        self.t_win, self.t_label = sliding_wd(self.scaled_data, size=30)
        # print(self.t_win[0])
        self.iter = len(self.t_label)  # number of windows
        print("Train data length:", self.iter)
        # Unsupervised scores
        self.scores = np.expand_dims(run_iforest(self.X_train), axis=1)  # score is array
        self.scores = self.scores[self.wnd_len - 1:]  # 29 = (window_size -1)
        self.scores = flatten(self.scores)
        # plt.plot(range(len(self.scores)), self.scores)
        # print(len(self.scores))

    def step(self, action):

        try:
            self.state = self.t_win[self.i + 1]
        except:
            self.state = self.t_win[self.i]
        alpha = 0.1
        beta = 1
        # Calculate reward
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if (operator.or_(self.t_label[self.i] == 0, self.t_label[self.i] == 1)):  # label available
            if action == 0:
                if self.t_label[self.i] == 0:
                    r = alpha
                else:
                    r = -alpha
            if action == 1:
                if self.t_label[self.i] == 1:
                    r = beta
                else:
                    r = -beta

        else:  # as the label is 0 or 1, for absent data this will be true
            if action == 0:
                if self.scores[self.i] == 0:
                    r = alpha
                else:
                    r = -alpha
            if action == 1:
                if self.scores[self.i] == 0:
                    r = -beta
                else:
                    r = beta

        self.i += 1  # i is the window_index, maximum of which will be self.iter

        if self.i >= self.iter:
            done = True
        else:
            done = False

        info = {}

        return self.state, r, done, info

    def render(self):
        pass

    def reset(self):
        self.i = 0
        self.done = False
        self.state = self.t_win[self.i]
        return self.state
