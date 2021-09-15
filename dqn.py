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
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

NUM_EPISODES = 5000
REPLAY_MEMORY_SIZE = 10000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
NUM_ACTIONS = 4
TARGET_UPDATE = 10

BATCH_SIZE = 128
HIDDEN_LAYER_SIZE = 256
GAMMA = .999
class Agent:
    def __init__(self):
        self.policy_net = DQN(N*N, HIDDEN_LAYER_SIZE, 4)
        self.target_net = DQN(N*N, HIDDEN_LAYER_SIZE, 4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-self.steps_done / EPS_DECAY)
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
                                           if s is not None])
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
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_model(self):
        for i_episode in range(NUM_EPISODES):
            # Initialize the environment and state
            game = Snake()
            state = torch.tensor(game.board.flatten(), dtype=torch.float)
            for t in count():
                # Select and perform an action
                action_idx = self.select_action(state)
                action = [UP, DOWN, LEFT, RIGHT][action_idx]
                game.run_step(action)
                self.steps_done += 1
                reward = torch.tensor([game.score])

                # Observe new state
                if not game.finished:
                    next_state = torch.tensor(game.board.flatten(), dtype=torch.float)
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
                    print(i_episode, game.score, t+1)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    a = Agent()
    a.train_model()

