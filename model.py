import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, layer_number, output_size, file=None):
        super().__init__()
        self.norm0 = nn.LayerNorm(input_size)
        self.embed_layer = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear_lat1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layer_number)])
        self.linear_lat2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(layer_number)])
        self.head_layer = nn.Linear(hidden_size, output_size)

        if file:
            self.load(file)

    def forward(self, x):

        x = self.embed_layer(x)
        for layer1,layer2 in zip(self.linear_lat1,self.linear_lat2):
            x = self.norm1(x)
            x = F.gelu(layer1(x))
            x = layer2(x)
        x = self.norm1(x)
        x = self.head_layer(x)
        return F.log_softmax(x, dim=-1)
    
    def save(self,file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)
    
    def load(self,file_name):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path,file_name)
        self.load_state_dict(torch.load(file_name))

class QTrainer:
    def __init__(self,model,lr,gamma,file = None):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        # (n, x)
        # n = 1
        if state.ndim == 1:
            # return
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()
