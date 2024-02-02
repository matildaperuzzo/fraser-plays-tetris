import argparse

import torch
import random
import numpy as np
from gameTetris import TetrisAI
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, file=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.total_cleared_lines = 0
        self.gamma = 0.9  # discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.file = file

        # num of states, hidden layer size, num of actions
        self.model = Linear_QNet(100, 512, 4, 4, file=self.file).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # choices are simple, medium, full
        self.method = "full"

    def get_state(self, game: TetrisAI) -> torch.Tensor:
        """
        State:
        # How many of the shape blocks fit into the placed blocks perfectly (number of blocks in shape)
        # wall left, wall right
        # distance from the bottom

        """

        method = self.method
        self.total_cleared_lines = game.total_cleared_lines

        if method == "medium":
            #find unique x values

            xPoints = game.x[np.argsort(game.y)]
            yPoints = game.y[np.argsort(game.y)]
            unique_x, indices = np.unique(xPoints, return_index=True)
            unique_y = yPoints[indices]
            unique_x = unique_x[np.argsort(unique_x)]
            unique_y = unique_y[np.argsort(unique_x)]

            unique_y[unique_y>game.height] = game.height

            top = np.zeros(game.width)+game.height
            top[unique_x] = unique_y

            bottom = np.zeros(game.width)
            for i in range(game.width):
                v = game.placedBlocks[i]
                ind = np.where(v == 1)[0]
                if len(ind) == 0:
                    ind = 0
                else:
                    ind = ind[-1]
                bottom[i] = ind
            bottom = bottom
            state = np.concatenate((top,bottom))
                

        if method == "simple":
            x = game.x
            y = game.y
            game.placedBlocks[x]

        if method == "full":
            state = game.state

        return state.flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_state, done = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_state,done) # train our model on this mini sample

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        log_probs = self.model(state)

        # tradeoff exploration / exploitation
        exploit = random.randint(0,1) < self.epsilon

        if self.total_cleared_lines % 100 == 0 and self.total_cleared_lines != 0:
            name = f"model_gamma{self.gamma}_lr{LR}_method-{self.method}.pth"
            self.model.save(file_name=name)

        return torch.multinomial(
            log_probs.exp() if exploit else torch.ones_like(log_probs),
            num_samples=1).squeeze(dim=-1)


def train(file = None, ui:bool = True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(file=file)
    game = TetrisAI(ui=ui)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        game.play_step(final_move)
        state_new = agent.get_state(game)
        if game.reward != 0:
            print(game.reward)  # this is zero when the block is falling

            # train short memory
            agent.train_short_memory(state_old, final_move, game.reward, state_new, game.done)

        # remember
        agent.remember(state_old, final_move, game.reward, state_new, game.done)

        if game.done:
            agent.n_games += 1
            # train long memory, plot result
            #agent.train_long_memory()

            if game.score > record:
                record = game.score
                agent.model.save()
            
            print(f'Game {agent.n_games} Score {game.score} Record {record}')

            plot_scores.append(game.score)
            total_score += game.score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            game.reset()

def get_distance_count(self,shape, game):
    #find points in shape different x values, selecting the lowest point
    yPoints = []
    xPoints = []
    for point in shape:
        if point.x not in xPoints:
            xPoints.append(point.x)
            yPoints.append(point.y)
        else:
            if point.y < yPoints[xPoints.index(point.x)]:
                yPoints[xPoints.index(point.x)] = point.y
    distances = []
    for x,y in zip(xPoints,yPoints):
        if game.placedBlocks[int(x//game.block_size)][int(y//game.block_size)] == 1:
            raise Exception('Not allowed to place shape here')
        v = game.placedBlocks[int(x//game.block_size)]
        ind = np.where(v == 1)[0]
        if len(ind) == 0:
            ind = 0
        else:
            ind = ind[-1]
        distances.append(int(y//game.block_size)-(ind))                 

    #count how many distances are the same as the smallest distance
    distances = np.array(distances)
    count = 0
    for d in distances:
        if d == min(distances):
            count += 1

    count_of_most_common = count/len(yPoints)
    return count_of_most_common, min(distances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a tetris agent")
    parser.add_argument("--file", type=str, help="File to load model from")
    parser.add_argument("--ui", action="store_true", help="Use the UI")

    train(**vars(parser.parse_args()))
