import gym
import json
import datetime as dt

from envs.trade_env import TRADEEnv

from stable_baselines3 import PPO

env = TRADEEnv("TSLA","2019-01-01", "2021-01-01", 0.00)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000)

# model = PPO("MlpPolicy", env, verbose=1)
# print("training started")
# model.learn(total_timesteps=1)
# print("training finished")
env1 = TRADEEnv("TSLA","2021-01-01", "2022-06-01", 0.00)
obs = env1.reset()
for i in range(15000):
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, done, info = env1.step(action)
    # print(obs) 
    env1.render()
    if done:
    	break
