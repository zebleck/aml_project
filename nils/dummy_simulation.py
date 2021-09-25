import numpy as np


states_normal = np.load("states_normal.npy")
states_lockdown = np.load("states_lockdown.npy")

class DummySimulation:
    def __init__(self):
        self.i = -1

    def update(self, action):
        self.i += 1
        if action == "lockdown":
            return states_lockdown[self.i]
        else:
            return states_normal[self.i]
