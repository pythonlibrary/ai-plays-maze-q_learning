from os import system
import time
import gym
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0',map_name="8x8",is_slippery=False)

def render_env(env):
    system('clear')
    env.render(mode='human')

with open('q_table_saved.pkl', 'rb') as f:
    q_table = pickle.load(f)

done=False
observation = env.reset()
while not done:
    action = np.argmax(q_table[observation])
    observation, reward, done, info = env.step(action)
    render_env(env)
    time.sleep(0.5)

env.close()

