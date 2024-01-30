import torch
import random
import numpy as np
from gameTetris import TetrisAI, Actions, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self,file = None):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.n_cleared_lines = 0
        self.gamma = 0.9 # discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.file = file
        self.model = Linear_QNet(12, 512, 512, 4, file = self.file) #num of states, hidden layer size, num of actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.method = "medium" #choices are simple, medium, full

    def get_state(self, game):
        """
        State:
        # How many of the shape blocks fit into the placed blocks perfectly (number of blocks in shape)
        # wall left, wall right
        # distance from the bottom

        """
        state = []

        method = self.method
        self.n_cleared_lines = game.n_cleared_lines

        if method == "medium":
            yPoints = []
            xPoints = []
            for point in game.shape:
                if point.x not in xPoints:
                    xPoints.append(point.x)
                    yPoints.append(point.y)
                else:
                    if point.y < yPoints[xPoints.index(point.x)]:
                        yPoints[xPoints.index(point.x)] = point.y
            #get shape x,y coordinates
            shapePos = [int(game.h//game.block_size) for i in range(int(game.w//game.block_size))]
            for x,y in zip(xPoints,yPoints):
                shapePos[int(x//game.block_size)] = y//game.block_size

            for i in shapePos:
                state.append(i)

            for i in range(int(game.w//game.block_size)):
                col = game.placedBlocks[i]
                #find location of highest block in column
                ind = np.where(col == 1)[0]
                if len(ind) == 0:
                    ind = 0
                else:
                    ind = ind[-1]
                state.append(ind)
        if method == "simple":
            try:
                shapeNothing = []
                for point in game.shape:
                    x = point.x
                    y = point.y - game.block_size
                    shapeNothing.append(Point(x,y))
                count_of_most_common, min_dist = get_distance_count(self,shapeNothing, game)
                state.append(count_of_most_common)
                state.append(min_dist)
                state.append(game.centerPoint.y//game.block_size)
            except:
                state.append(0)
                state.append(0)
                state.append(0)

            shapeLeft = []
            for point in game.shape:
                x = point.x - game.block_size
                y = point.y - game.block_size
                shapeLeft.append(Point(x,y))
            
            try:
                count_of_most_common, min_dist = get_distance_count(self,shapeLeft, game)
                state.append(count_of_most_common)
                state.append(min_dist)
                state.append(game.centerPoint.y//game.block_size)                    
                
            except:
                state.append(0)
                state.append(0)
                state.append(0)

            shapeRight = []
            for point in game.shape:
                x = point.x + game.block_size
                y = point.y - game.block_size
                shapeRight.append(Point(x,y))
            
            try:
                count_of_most_common, min_dist = get_distance_count(self,shapeRight, game)
                state.append(count_of_most_common)
                state.append(min_dist)
                state.append(game.centerPoint.y//game.block_size)

            except:
                state.append(0)
                state.append(0)
                state.append(0)

            shapeRotate = game.rotate_shape(game.shape, (game.centerPoint.x, game.centerPoint.y))
            for i,point in enumerate(shapeRotate):
                x = point.x
                y = point.y - game.block_size
                shapeRotate[i] = Point(x,y)

            try:
                count_of_most_common, min_dist = get_distance_count(self,shapeRotate, game)
                state.append(count_of_most_common)
                state.append(min_dist)
                state.append(game.centerPoint.y//game.block_size)

            except:
                state.append(0)
                state.append(0)
                state.append(0) 

        if method == "full":
            # find distance between placed blocks and shape
            for point in game.shape:
                #find highest point in placed blocks with same x
                highest = 0
                for i in range(len(game.placedBlocks)):
                    if game.placedBlocks[i][int(point.x//game.block_size)] == 1:
                        highest = i

            for x,y in game.shape:
                state.append(point.x)
                state.append(point.y)
            for i in game.placedBlocks:
                for j in i:
                    state.append(j)

        state = np.array(state)
        return state

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 10000 - self.n_cleared_lines
        final_move = [0, 0, 0, 0]
        if random.randint(0, 50000) < self.epsilon and self.file == None:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        if self.n_cleared_lines % 100 == 0 and self.n_cleared_lines != 0:
            name = f"model_gamma{self.gamma}_lr{LR}_method-{self.method}.pth"
            self.model.save(file_name=name)
        return final_move


def train(file = None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(file = file)
    game = TetrisAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

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
    train(file="model_gamma0.9_lr0.001_method-simple.pth")
    # train()