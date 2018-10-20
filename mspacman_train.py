import os

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor

from utils import create_logger, create_model_checkpoint


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
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_actions, activation='linear'))
model.summary()

NUM_STEPS = 10000

memory = SequentialMemory(limit=round(0.75 * NUM_STEPS), window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05,
                              nb_steps=round(0.8 * NUM_STEPS))
test_policy = EpsGreedyQPolicy(eps=0.05)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, processor=ScoreProcessor(),
               target_model_update=1e-2, policy=policy, test_policy=test_policy)
dqn.compile(Adam(lr=5e-4), metrics=['mae'])

tensorboard = create_logger(ENV_NAME)
checkpoint_dir, checkpoint = create_model_checkpoint(ENV_NAME, 'episode_reward', 'max', save_best_only=True,
                                                     save_weights_only=True)

with open(os.path.join(checkpoint_dir, 'model.json'), 'w') as f:
    f.write(model.to_json())

dqn.fit(env, nb_steps=NUM_STEPS, visualize=False, verbose=1, callbacks=[tensorboard, checkpoint])

dqn.test(env, nb_episodes=15, visualize=True, verbose=1, nb_max_start_steps=100)

model.save('models/dqn_{}.h5'.format(ENV_NAME))