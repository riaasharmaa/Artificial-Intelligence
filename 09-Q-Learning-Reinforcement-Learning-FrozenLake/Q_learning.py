import gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0
def default_dict_factory():
    return defaultdict(default_Q_value)

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_dict_factory)
    episode_reward_record = deque(maxlen=100)

    MAX_STEPS_PER_EPISODE = 100

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                action = max(Q_table[obs], key=Q_table[obs].get)
    
            #apply action to environment
            next_obs, reward, done, terminated, info = env.step(action)
            episode_reward += reward

            #next_obs -> integer
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            #initialize if not present
            if next_obs not in Q_table:
                Q_table[next_obs] = defaultdict(default_Q_value)

            #update Q-table
            if not done:
                max_next_action_value = max(Q_table[next_obs].values(), default=default_Q_value())
                Q_table[obs][action] = (1 - LEARNING_RATE) * Q_table[obs][action] + \
                                       LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_action_value)
            else:
                Q_table[obs][action] = (1 - LEARNING_RATE) * Q_table[obs][action] + LEARNING_RATE * reward

            #update obs
            obs = next_obs

            if done or steps >= MAX_STEPS_PER_EPISODE:
                break

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)
        EPSILON *= EPSILON_DECAY
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )

    standard_dict_Q_table = {state: dict(actions) for state, actions in Q_table.items()}
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################
