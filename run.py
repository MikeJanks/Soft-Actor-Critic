import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import numpy as np
import time

from SAC import SAC

from tools.envs import Atari as Env
# from tools.envs import Pong, Mario

# env_name = "LunarLanderContinuous-v2"
# env_name = "LunarLander-v2"
# env_name = "BipedalWalkerHardcore-v2"
# env_name = "BipedalWalker-v2"
# env_name = "CartPole-v0"
env_name = "Pong-v0"
# env_name = "MountainCar-v0"

# env = Env(env_name)
env = Env(env_name, do_preprocess=True, skips=4)
# sac = SAC(env.action_space.shape[0], env.observation_space)
sac = SAC(env.action_space.n, env.observation_space)
running_reward=None
steps=0
for e in range(100000):
    s = env.reset()
    ep_score = 0
    done = False
    
    while not done:
        a = sac.action(s)
        
        # n_s, r, done, info = env.step(a)
        n_s, r, done, info = env.step(np.argmax(a))
        sac.add(s, a, r, n_s, done)
        sac.update()
        
        s = n_s
        ep_score+=r
        steps+=1

    print(sac.log_alpha.numpy())
    print(sac.alpha.numpy())

    running_reward = ep_score if running_reward==None else running_reward * 0.99 + ep_score * 0.01
    print(f'SAC Episode: {e} | Steps: {steps} | Episode Reward: {ep_score} | Average Reward: {running_reward}')