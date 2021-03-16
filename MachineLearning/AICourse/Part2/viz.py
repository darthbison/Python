import os
import cv2
import glob
import gym
import vizdoom
import matplotlib.pyplot as plt
from collections import Counter
from gym.wrappers import Monitor


env = gym.make('VizdoomBasic-v0')
action_num = env.action_space.n
print("Number of possible actions: ", action_num)
state = env.reset()
state, reward, done, info = env.step(env.action_space.sample())
print(state.shape)
env.close()


observation = env.reset()
plt.imshow(observation)
plt.show()
print(observation.shape)


