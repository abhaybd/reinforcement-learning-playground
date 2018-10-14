from keras.models import load_model
import gym
import numpy as np

model = load_model('models/dqn_Blackjack-v0_weights.h5')

ENV_NAME = 'Blackjack-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)


def debug(msg, enabled):
    if enabled:
        print(msg)


def play_game(debug_enabled=True):
    done = False
    score, dealer, usable = env.reset()
    reward = 0
    debug('Start!\nScore: %s, Dealer: %s, Usable Ace: %s' % (score, dealer, bool(usable)), debug_enabled)
    while not done:
        action = np.argmax(model.predict(np.expand_dims([[score, dealer, usable]], axis=0))[0])
        (score, dealer, usable), reward, done, _ = env.step(action)
        debug('\nAction: %s' % ('hit' if action == 1 else 'stick'), debug_enabled)
        debug('Score: %s, Dealer: %s, Usable Ace: %s' % (score, dealer, bool(usable)), debug_enabled)
    if reward < 0:
        debug('Lost!', debug_enabled)
    elif reward > 0:
        debug('Won!', debug_enabled)
    else:
        debug('Tie!', debug_enabled)
    return reward > 0, reward == 0, reward < 0


win_sum, tie_sum, lose_sum = 0, 0, 0
n_games = 10000
for i in range(n_games):
    result = play_game(False)
    win_sum += 1 if result[0] else 0
    tie_sum += 1 if result[1] else 0
    lose_sum += 1 if result[2] else 0

print('Win: %.2f, Tie: %.2f, Lose: %.2f' % (100 * win_sum / n_games, 100 * tie_sum / n_games, 100 * lose_sum / n_games))
