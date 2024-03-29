import random
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


def plot(scores, mean_scores, game_rewards = None):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)

    if game_rewards:
        plt.plot(game_rewards)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.savefig('plot.png')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class Memory(Dataset):
    def __init__(self, max_size = 1000):
        self.memory = []
        self.max_size = max_size
    
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index):
        return self.memory[index]
    
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    # def get_samples(self, batch_size=32, shuffle=True):

        # batch = DataLoader(self, batch_size=batch_size, shuffle=shuffle)

        # for state, action, reward, next_state, done in batch:
        #     return state,action,reward,next_state,done
