from snake import Snake, UP, DOWN, LEFT, RIGHT, N

from collections import namedtuple, deque
from itertools import count
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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


class DQN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.linear1 = nn.Linear(64, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print("***")
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        # x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


NUM_EPISODES = 50000
REPLAY_MEMORY_SIZE = 50000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2e-6
NUM_ACTIONS = 4
TARGET_UPDATE = 1000

BATCH_SIZE = 32
HIDDEN_LAYER_SIZE = 256
GAMMA = .97


class Agent:
    def __init__(self):
        self.policy_net = DQN(HIDDEN_LAYER_SIZE, 4)
        self.target_net = DQN(HIDDEN_LAYER_SIZE, 4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=.001)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.episode_durations = []

        self.scores = []

    def select_action(self, state):
        eps_threshold = max(EPS_START - self.steps_done * EPS_DECAY, EPS_END)
        if np.random.random() > eps_threshold:
            prediction = self.policy_net(state)
            action = torch.argmax(prediction).item()
        else:
            action = torch.tensor([[random.randrange(4)]], dtype=torch.long)

        return action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                             if s is not None])[:, None, :, :]

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)[np.arange(state_batch.shape[0]), action_batch]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_model(self):
        for i_episode in range(NUM_EPISODES):
            # Initialize the environment and state
            old_score = 0
            game = Snake()
            state = torch.tensor(game.board, dtype=torch.float)

            for t in count():
                state = torch.unsqueeze(state, 0)

                # Select and perform an action
                action_idx = self.select_action(torch.unsqueeze(state, 0))
                action = [UP, DOWN, LEFT, RIGHT][action_idx]
                game.run_step(action)
                self.steps_done += 1
                reward = torch.tensor([game.score - old_score])
                old_score = game.score

                # Observe new state
                if not game.finished:
                    next_state = torch.tensor(game.board, dtype=torch.float)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action_idx, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if game.finished:
                    self.episode_durations.append(t + 1)
                    self.scores.append(game.score)
                    string = f"Episode: {i_episode}, Score: {game.score}"
                    if len(self.scores) >= 1000:
                        string += f", Mean Score: {round(np.sum(self.scores[-1000:]) / 1000, 2)}    "
                    print("\r{}".format(string), end="")
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        np.save("snake_scores.npy", self.scores)


if __name__ == "__main__":
    a = Agent()
    a.train_model()

