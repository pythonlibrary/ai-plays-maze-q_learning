from os import system
import time
import gym

env = gym.make('FrozenLake-v0', map_name='8x8', is_slippery=False)


print()
print("Observation Space")
print(env.observation_space)
print("Action Space")
print(env.action_space)

done = False
observation = env.reset()

print()
print("Maze Reset")
env.render()

observation, reward, done, info = env.step(1)
print("Robot moved by one step(down)")
env.render()

print()
print(f"observation={observation}")
print(f"reward={reward}")
print(f"done={done}")


env.close()