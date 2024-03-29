import argparse
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from tetris_school.games import Tetris
from tetris_school.model import Fraser, Jordan, JordanSimple
from tetris_school.utils import plot, Memory, DataLoader

import random

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

def train(learning_rate: float = 0.001, temperature: float = 1.0, ui: bool = True, file: Optional[str] = None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = Memory()
    gamma = 0.99

    plot_scores = []
    plot_mean_scores = []
    plot_game_rewards = []
    n_games = 0
    total_score = 0
    record = 0

    game = Tetris(ui=ui)
    model = Jordan(hidden_size=512, layer_number=2, input_size=2*game.state.shape[0], num_actions=4).to(device)
    # model = JordanSimple(input_size = 2*game.state.shape[0], hidden_size = 512, output_size = 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss = torch.tensor(0, dtype=torch.float, device=device)

    while True:
        state = get_state(game)
        actions = model(state)
        # actions = F.softmax(choices, dim=0)
        # exploration-exploitation trade-off
        epsilon = 100 - game.total_cleared_lines

        if random.randint(0, 100) < epsilon:
            action = random.randint(0, 2)
        else:
            action = torch.argmax(actions).item()
        game.play_step(action)
        next_state = get_state(game)
        reward = game.reward
        done = game.done

        memory.remember(state, action, reward, next_state, done)
        train_step(model, state, next_state, actions, reward, done, memory, optimizer, criterion, loss, gamma)
        

        if game.done:
            n_games += 1
            print(f'Game {n_games} Score {game.score} Record {record}')

            plot_scores.append(game.score)
            if game.score > record:
                record = game.score

            total_score += game.score
            mean_score = total_score / n_games
            plot_game_rewards.append(game.game_reward)
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)
            temperature *= 0.999

            game.reset()

def train_step(model, state, next_state, action , reward, done, memory, optimizer, criterion, loss, gamma):
    if not done:
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.tensor([reward])
        done = torch.tensor([done])
    else:
        #sample from memory
        if len(memory) < BATCH_SIZE:
            mini_sample = next(iter(DataLoader(memory, batch_size=len(memory), shuffle=True)))
        else:
            while True:
                try:
                    mini_sample = next(iter(DataLoader(memory, batch_size=BATCH_SIZE, shuffle=True)))
                    break
                except:
                    continue
        state = mini_sample[0]
        action = mini_sample[1]
        reward = mini_sample[2]
        next_state = mini_sample[3]
        done = mini_sample[4]
            
    pred = model(state)
    target = pred.clone()
    for idx in range(len(done)):
        Q_new = reward[idx]
        if not done[idx]:
            Q_new = reward[idx] + gamma * torch.max(model(next_state[idx]))

        target[idx][torch.argmax(action[idx]).item()] = Q_new
        
    
    loss = criterion(pred, target)
    if len(done) > 1:
        print(loss.item())
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


def get_state(game):
    x = game.state
    floor = torch.zeros(x.shape[0]).int()
    cieling = torch.zeros_like(floor)
    floor_index = torch.argwhere(x == 1).int()
    floor[floor_index[:,0]] = floor_index[:,1]+1
    cieling_index = torch.argwhere(x == 2).int()
    cieling[cieling_index[:,0]] = cieling_index[:,1]+1
    state = torch.cat([floor,cieling])
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a tetris agent")
    parser.add_argument("--file", type=str, help="File to load model from")
    parser.add_argument("--ui", action="store_true", help="Use the UI")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for action sampling")

    train(**vars(parser.parse_args()))
