import os
from datetime import datetime

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor


class ScoreProcessor(Processor):
    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


ENV_NAME = 'MsPacman-ram-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

WINDOW_LENGTH = 4

model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH, 128)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_actions, activation='linear'))
model.summary()

memory = SequentialMemory(limit=1500000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1800000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, processor=ScoreProcessor(),
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

date = datetime.today().strftime('%Y-%m-%d_%H%M')
log_dir = 'logs/%s/%s' % (ENV_NAME, date)
os.makedirs(log_dir)

tensorboard = TensorBoard(log_dir=log_dir)

dqn.fit(env, nb_steps=2500000, visualize=False, verbose=1, callbacks=[tensorboard])

dqn.test(env, nb_episodes=10, visualize=True, verbose=1)

model.save('models/dqn_{}.h5'.format(ENV_NAME))
