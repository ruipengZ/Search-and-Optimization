### Deep Q-learning on catching block game
import numpy as np 
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

gamma = 0.9
eps_high = 0.9
eps_low = 0.1
num_episodes = 500
LR = 0.001
batch_size = 5
decay = 200

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        board = np.zeros((10, 10))
        board[9, 2:7] = 1.
        self.state = board
        self.total_steps = 0
        self.score = 0
        self.done = False
        return self.state[np.newaxis, :].copy()

    def step(self, action):
        reward = 0.
        board = self.state

        if self.done:
            print('Call step after env is done')

        if self.total_steps == 200:
            self.done = True
            return board[np.newaxis, :].copy(), 10, self.done

        # take actions
        # move to left
        if action == 0 and board[9][0] != 1:
            board[9][0:9] = board[9][1:10].copy()
            board[9][9] = 0
        # move to right
        elif action == 1 and board[9][7] != 1:
            board[9][1:10] = board[9][0:9].copy()
            board[9][0] = 0
        # whether the block drop to the ground or the plate
        if 1 in board[8]:
            block = np.where(board[8] == 1)
            if board[9][block] != 1:
                reward = -1.
                self.done = True  # drop on the ground - game over
            else:
                reward = 1.
                self.score += 1
            board[8][block] = 0.
        # the block falls for 1
        board[1:9] = board[0:8].copy()
        board[0] = 0

        if self.total_steps % 8 == 0:
            idx = random.randint(0, 9) # a new block is born
            board[0][idx] = 1.
        self.total_steps += 1

        return board[np.newaxis, :].copy(), reward, self.done


class DQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DQLearning():
    def __init__(self, network, eps, gamma, lr):
        self.network = network
        self.eps = eps
        self.gamma = gamma
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def learn(self, batch):
        s0, a0, r1, s1, done = zip(*batch)

        n = len(s0)
        s0 = torch.FloatTensor(s0)
        s1 = torch.FloatTensor(s1)
        r1 = torch.FloatTensor(r1)
        a0 = torch.LongTensor(a0)
        done = torch.BoolTensor(done)

        y_true = r1 + self.gamma * torch.max(self.network(s1).detach(), dim=1)[0]
        y_pred = self.network(s0)[range(n), a0]

        loss = F.mse_loss(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pick_action(self, state):
        # epsilon greedy
        state = state[np.newaxis, :]
        action_value = self.network(torch.FloatTensor(state))

        if random.random() < self.eps:
            return random.randint(0, 2)
        else:
            max_action = torch.argmax(action_value, dim=1)
            return max_action.item()

### Deep Q learning ###
net = DQNet()
agent = DQLearning(net, 1, gamma, LR)
game = Game()
replay_mem = deque(maxlen=5000)

state = game.reset()

for episode in range(num_episodes):
    agent.eps = eps_low + (eps_high - eps_low) * (np.exp(-1.0 * episode / decay))
    s0 = game.reset()
    while True:
        a0 = agent.pick_action(s0)
        s1, r1, done = game.step(a0)
        replay_mem.append((s0.copy(), a0, r1, s1.copy(), done))

        if done:
            break

        s0 = s1

        if replay_mem.__len__() >= batch_size:
            batch = random.sample(replay_mem, k=batch_size)
            loss = agent.learn(batch)

## Display the game ###
ims=[]
frames = []
fig = plt.figure()
plt.axis('off')
plt.tight_layout()

agent.eps = 0
s = game.reset()
while True:
    a = agent.pick_action(s)
    s, r, done = game.step(a)

    if done:
        break

    img = s.squeeze()
    frame = plt.imshow(img)
    frames.append(frame)

    plt1 = plt.text(-1, 0, "Score: %d" % (game.score), fontsize=15,
                    bbox={'fc': 'w', 'ec': 'k'})
    frames.append(plt1)

    ims.append(frames.copy())

ani = animation.ArtistAnimation(fig, ims, interval=1)
writer = PillowWriter(fps=5)
ani.save("./gif/D_QL.gif", writer=writer)
