"""
Auxiliary methods for plotting the simulations
"""

import numpy as np
import pandas as pd

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *

legend_ecom = {
    'Q1': 'Most Poor', 'Q2': 'Poor', 'Q3': 'Working Class',
    'Q4': 'Rich', 'Q5': 'Most Rich', 'Business':'Business', 'Government':'Government'
}
"""Legend for wealth distribution quintiles"""


def update_statistics(sim, statistics):
    """Store the iteration statistics"""

    stats1 = sim.get_statistics(kind='info')
    statistics['info'].append(stats1)

    stats2 = sim.get_statistics(kind='ecom')
    statistics['ecom'].append(stats2)

def update(sim, statistics):
    """
    Execute an iteration of the simulation and update the animation graphics

    :param sim:
    :param scat:
    :param linhas1:
    :param linhas2:
    :param statistics:
    :return:
    """
    sim.execute()

    update_statistics(sim, statistics)


def execute_simulation_step(sim, action):
    sim.execute()
    state = sim.get_statistics(kind='info')

    return state

def execute_simulation(sim, **kwargs):
    """
    Execute a simulation and plot its results

    :param sim: a Simulation or MultiopulationSimulation object
    :param iterations: number of interations of the simulation
    :param  iteration_time: time (in miliseconds) between each iteration
    :return: an animation object
    """
    statistics = {'info': [], 'ecom': []}

    iterations = kwargs.get('iterations', 100)

    sim.initialize()

    update_statistics(sim, statistics)

    for i in range(iterations):
        update(sim, statistics)
        print("*"*50)
        print(i)
        print(sim.get_statistics(kind='info'))

    return statistics