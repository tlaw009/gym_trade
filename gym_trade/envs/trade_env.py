import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import logging
import numpy as np
import math, sys
import yfinance as yf
import re


# MAX_ACCOUNT_BALANCE = 2147483647
# MAX_NUM_SHARES = 2147483647
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 200000


# def padding(array, xx, yy):
#     """
#     :param array: numpy array
#     :param xx: desired height
#     :param yy: desirex width
#     :return: padded array
#     """

#     h = array.shape[0]
#     w = array.shape[1]

#     a = (xx - h) // 2
#     aa = xx - a - h

#     b = (yy - w) // 2
#     bb = yy - b - w

#     return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


class TRADEEnv(gym.Env, utils.EzPickle):
    def __init__(self,ticker_name, start=None, end=None, commission_rate = 0.00):
        self.r = re.compile('\d{4}-\d{2}-\d{2}')
        self.ticker = yf.Ticker(ticker_name)
        self.done = False
        self.c_r = commission_rate
        self.total_possible = 0
        # self.loss_lower_bound = -100
        self.look_back = False
        if not start == None and not end == None:
            if self.r.match(start) and self.r.match(end):
                self.look_back = True
                self.data = yf.download(ticker_name, start = start, end = end, group_by = "ticker")
                self.aux_data = yf.download("^VIX", start = start, end = end, group_by = "ticker")
                if len(self.data.to_numpy()) < 30:
                    raise ValueError("period too short, please select period of sufficient length")
                self.env_step_index = 1
                self.env_step_end_index = len(self.data.to_numpy())-1
        else:
            print("No valid period input received, real time environment is initialized", flush=True)


        # [[buy, sell, hold], [% of balance, % of shares, meaningless stuff]]
        self.action_space = spaces.Box(
        low=np.array([-1]), high=np.array([1]), dtype=np.float32)


        self.observation_space = spaces.Box(
        low=0, high=1000000000000000, shape=(8, ), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_possible = 0
        if self.look_back:
            self.env_step_index = 1
        self.done = False
        # self.loss_lower_bound = -100

        # print("env reset")
        return self._next_observation()

    def _next_observation(self):
        if self.look_back:
            obs= np.concatenate((self.data.to_numpy()[self.env_step_index][0].flatten()/1000.0,
                                        self.data.to_numpy()[self.env_step_index][0].flatten()/1000.0 - self.data.to_numpy()[self.env_step_index-1][0].flatten()/1000.0,
                                        self.data.to_numpy()[self.env_step_index][5].flatten()/1000000.0,
                                        self.aux_data.to_numpy()[self.env_step_index][0].flatten()/1000.0,
                                        # np.delete(self.data.to_numpy()[self.env_step_index-5: self.env_step_index-1], 4, 1).flatten(),
                                        # np.delete(self.aux_data.to_numpy()[self.env_step_index-5: self.env_step_index-1], 4, 1).flatten(),
                                        # padding(self.ticker.balance_sheet.to_numpy(), 26, 5),
                                        np.array([[self.net_worth/1000000.0, self.balance/1000000.0, self.shares_held/1000000.0, self.cost_basis/1000000.0]]).flatten()), axis=None)
            # print(obs.flatten())
            return obs.flatten()
        else:
            obs = np.concatenate((self.ticker.history(period="1d", interval="1m").to_numpy()[-2,0].flatten()/1000.0,
                                        self.ticker.history(period="1d", interval="1m").to_numpy()[-2,0].flatten()/1000.0 - self.ticker.history(period="1d", interval="1m").to_numpy()[-3,0].flatten()/1000.0,
                                        self.ticker.history(period="1d", interval="1m").to_numpy()[-3,4].flatten()/1000000,
                                        yf.Ticker("^VIX").history(period="5d", interval="1d").to_numpy()[-2,0].flatten()/1000.0,
                                        # np.delete(self.ticker.history(period="5d", interval="1d").to_numpy(), [-2,-1], 1).flatten(),
                                        # np.delete(yf.Ticker("^VIX").history(period="5d", interval="1d").to_numpy(), [-2,-1], 1).flatten(),
                                        # padding(self.ticker.balance_sheet.to_numpy(), 26, 5),
                                        np.array([[self.net_worth/1000000.0, self.balance/1000000.0, self.shares_held/1000000.0, self.cost_basis/1000000.0]]).flatten()), axis = None)
            return obs.flatten()


    def _take_action(self, action):
        # Set the current price to a random price within the time step
        if self.look_back:
            current_price = self.data.to_numpy()[self.env_step_index][0]
            next_price = self.data.to_numpy()[self.env_step_index+1][0]
        else:
            current_price = self.ticker.history(period="1d", interval="1m").to_numpy()[-2,0]
            next_price = self.ticker.history(period="1d", interval="1m").to_numpy()[-1,0]
        self.total_possible = int(self.balance / (current_price*(1+self.c_r)))
        amount = np.abs(action)
        action_cost = 0
        if action > 0.0:
            # Buy amount % of balance in shares
            shares_bought = int(self.total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price
            action_cost = additional_cost*self.c_r
            # print("spent: ", action_cost)
            self.balance -= additional_cost
            if self.shares_held + shares_bought > 0:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought < 1:
                if next_price > current_price:
                    gain = (self.shares_held - self.total_possible)*(next_price-current_price)
                else:
                    gain = self.shares_held*(next_price-current_price)
            else:
                gain = (shares_bought - self.total_possible + self.shares_held)*(next_price-current_price)



        elif action < 0.0:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
            action_cost = (shares_sold* current_price)*self.c_r

            if shares_sold < 1:    
                if next_price > current_price:
                    gain = (self.shares_held - self.total_possible)*(next_price-current_price)
                else:
                    gain = self.shares_held*(next_price-current_price)
            else:
                gain = (shares_sold - self.shares_held + self.total_possible)*(current_price-next_price)

        else:
            if next_price > current_price:
                gain = (self.shares_held - self.total_possible)*(next_price-current_price)
            else:
                gain = self.shares_held*(next_price-current_price)


        #     print("gained: ", shares_sold* current_price)
        # print("Commission fee: ", action_cost)
        self.balance = self.balance - action_cost
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

        return gain


    def step(self, action):
        # Execute one time step within the environment
        gain = self._take_action(action)

        if self.look_back:
            if not self.done:
                self.env_step_index = self.env_step_index + 1
            if self.env_step_index == self.env_step_end_index - 1:
                self.done = True

        reward = gain

        # if reward < self.loss_lower_bound:
        #     self.loss_lower_bound = reward

        # print(reward)
        if not self.done and self.net_worth <= 0:
            self.done = True

        obs = self._next_observation()

        return obs, reward, self.done, {}

    def render(self, close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Balance: {self.balance}', flush=True)
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})', flush=True)
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})', flush=True)
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})', flush=True)
        print(f'Profit: {profit}', flush=True)

# def local_test():
#     env = TRADEEnv("MSFT")
#     print(env.reset())

# local_test()