import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def plot(scores):
    plt.figure()

    plt.plot(scores, label='Score')
    plt.xlabel('Episode')
    plt.legend()

    plt.savefig("plot.png")
    plt.close()