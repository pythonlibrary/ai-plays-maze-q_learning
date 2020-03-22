from os import system
import time
import gym
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
STATS_EVERY = 100
SHOW_EVERY = 1000

epsilon = 1  
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES*7//10
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

env = gym.make('FrozenLake-v0',map_name="8x8",is_slippery=False)

def render_env(env):
    system('clear')
    env.render(mode='human')

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

q_table = np.random.uniform(low=-1, high=1, size=([env.observation_space.n, env.action_space.n]))

for episode in tqdm(range(EPISODES)):
    episode_reward = 0
    done = False

    observation = env.reset()
    while not done:
        # lookup q_table to get next action
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[observation])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_observation, reward, done, info = env.step(action)

        if episode % SHOW_EVERY == 0:
            render_env(env)
            time.sleep(0.1)

        episode_reward += reward

        if not done:
            max_future_q = np.max(q_table[new_observation])
            current_q = q_table[observation][action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[observation][action] = new_q
        else:
            q_table[observation][action] = reward 
            # if reward > 0:
            #     q_table[observation][action] = reward 
            # else:
            #     q_table[observation][action] = -1

        observation = new_observation

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards)/len(ep_rewards)
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards))
        aggr_ep_rewards['min'].append(min(ep_rewards))
        ep_rewards = []

    
env.close()

with open('q_table_saved.pkl', 'wb') as f:
    pickle.dump(q_table, f)


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend()
plt.show()
