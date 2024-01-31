# %% 
from keras.models import Sequential
from keras.layers import Dense
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import gym 
import random
# %% 
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, t, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))

# %%
model = Sequential()
model.add(Dense(24, input_shape=(states,), activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='softmax'))

model.summary()
# %%
