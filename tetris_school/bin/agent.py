import argparse
from typing import Optional
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from tetris_school.games import Tetris
from tetris_school.model import Fraser, Jordan, Anderson
from tetris_school.utils import plot, Memory, DataLoader

import random

MAX_MEMORY = 10_000
BATCH_SIZE = 128

class Agent():
    def __init__(self, learning_rate = 1e-4, gamma = 0.99, tau = 0.005, 
                 temp_annealing = 0.99, max_temp = 1, min_temp = 0.05, 
                 width = 5, height = 5,
                 file = "tetris_model.pth"):
        self.game = Tetris(ui=True, width=width, height=height)
        # self.model = Jordan(hidden_size=128, layer_number=4, input_size=2*self.game.width, num_actions=4)
        self.model = Anderson(hidden_size=64, layer_number=3, tot_size=self.game.width*self.game.height, num_actions=5)
        self.model_target = Anderson(hidden_size=64, layer_number=3, tot_size=self.game.width*self.game.height, num_actions=5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = Memory()
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.annealing = temp_annealing
        self.gamma = gamma
        self.tau = tau
        self.n_steps = 0
        self.file = file

    def train(self):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.n_games = 0
        self.record = 0
        self.total_score = 0
        for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(param.data)

        while True:

            state = self.game.state.clone()
            q_pred = self.model(state.unsqueeze(0))
            self.temperature = max(self.min_temp, self.max_temp * (self.annealing ** self.n_games))
            if random.random() < self.temperature:
                action_des = torch.zeros(5, dtype = torch.float)
                for i in range(5):
                    pw = 1-(self.game.height - self.game.y.min())/self.game.height+0.001
                    action_des[i] = self.game._check_move(torch.tensor(i, dtype=torch.int))
                action_des = F.softmax(action_des/pw, dim=-1)
                action = torch.multinomial(action_des, 1)[0]
            else:
                q_pred = F.softmax(q_pred, dim=-1)
                action = torch.multinomial(q_pred, 1)[0][0]
            self.temperature *= self.annealing
    
            self.game.play_step(action)
            reward = self.game.reward.clone()
            self.n_steps += 1
            new_state = self.game.state.clone()
            self.memory.remember(state, action, reward, new_state, self.game.done)

            # train from memory
            if len(self.memory)>BATCH_SIZE:
                print('Training from memory')    
                loader = DataLoader(self.memory, batch_size=BATCH_SIZE, shuffle=True)
                try:
                    states, actions, rewards, new_states, dones = next(iter(loader))
                except:
                    print('Error in loading data')
                    states, actions, rewards, new_states, dones = next(iter(loader))
                self.train_step(states, actions, rewards, new_states, dones)
                for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                    target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)


            # update game stats
            if self.game.done:
                self.n_games += 1
                if self.game.score > self.record:
                    self.record = self.game.score
                    self.save()
                self.total_score += self.game.score

                print(f'Game {self.n_games} Score {self.game.score} Record {self.record}')

                self.plot_scores.append(self.game.score)
                mean_score = self.total_score / self.n_games
                self.plot_mean_scores.append(mean_score)

                plot(self.plot_scores, self.plot_mean_scores)
                #train entire game
                # state, action, reward, new_state, done = self.memory.get_samples(self.n_steps, shuffle=True)
                self.n_steps = 0

                self.game.reset()



    def train_step(self, state, action, reward, new_state, done):

        if done.ndim == 0:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            new_state = torch.unsqueeze(new_state, 0)
            done = torch.unsqueeze(done, 0)

        predicted_q = self.model(state)

        with torch.no_grad():
            new_q = predicted_q.clone()
            for idx in range(len(done.tolist())):
                if done[idx]:
                    new_q[idx][action[idx]] = reward[idx]
                else:
                    new_q[idx][action[idx]] = (reward[idx] + self.gamma * torch.max(self.model_target(new_state[idx].unsqueeze(0))))

        loss = self.criterion(new_q,predicted_q)
        self.optimizer.zero_grad()
        loss.backward()
        # print(loss)
        #clip the gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()
        

    def get_state(self):
        blocks = self.game.placedBlocks.clone()
        #find floor
        floor = torch.zeros(self.game.width,dtype=torch.int)
        for i in range(self.game.width):
            if any(blocks[i]):
                floor[i] = torch.nonzero(blocks[i]).max()+1
        #find ceiling
        ceiling = torch.zeros_like(floor)
        ceiling[self.game.x] = self.game.y

        state = torch.cat([floor,ceiling])
        return state    
    
    def save(self):
        torch.save(self.model.state_dict(), self.file)
    
    def load(self, file = "tetris_model.pth"):
        self.model.load_state_dict(torch.load(file))

if __name__ == '__main__':
    agent = Agent(temp_annealing=0.999,width=6,height=12)
    agent.train()
