import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
from network.graph_abs import *
from network.util import *
from tqdm import tqdm


def sleep(a):
    if not new_day(a.iteration) and bed_time(a.iteration):
        return True
    #elif 9 <= a.iteration % 24 <= 11 and 14 <= a.iteration % 24 <= 16:
    #    return True
    return False


global_parameters = dict(

    # General Parameters
    length=300,
    height=300,

    # Demographic
    population_size=100,
    homemates_avg=3,
    homeless_rate=0.0005,
    amplitudes={
        Status.Susceptible: 10,
        Status.Recovered_Immune: 10,
        Status.Infected: 10
    },

    # Epidemiological
    critical_limit=0.01,
    contagion_rate=.9,
    incubation_time=5,
    contagion_time=10,
    recovering_time=20,

    # Economical
    total_wealth=10000000,
    total_business=9,
    minimum_income=900.0,
    minimum_expense=600.0,
    public_gdp_share=0.1,
    business_gdp_share=0.5,
    unemployment_rate=0.12,
    business_distance=20
)

def plot_statistics(statistics):
    x = list(range(len(statistics)))
    keys = list(statistics[0].keys())

    y_dict = {key: [] for key in keys}
    for i in statistics:
        for key in keys:
            y_dict[key].append(statistics[i][key])

    for key in y_dict:
        plt.plot(x, y_dict[key], label=key)

    plt.legend()
    plt.show()


def get_simulation():
    scenario0 = dict(
        name='scenario0',
        initial_infected_perc=.05,
        initial_immune_perc=.00,
        contagion_distance=1.,
    )
    sim = GraphSimulation(**{**global_parameters, **scenario0})
    sim.initialize()

    return sim

if __name__ == "__main__":
    num_steps = 1000

    scenario0 = dict(
        name='scenario0',
        initial_infected_perc=.05,
        initial_immune_perc=.00,
        contagion_distance=1.,
    )
    scenario1 = dict(
        name='scenario1',
        initial_infected_perc=.05,
        initial_immune_perc=.00,
        contagion_distance=1.,
        callbacks={'on_execute': lambda x: sleep(x)}
    )

    sim = GraphSimulation(**{**global_parameters, **scenario0})
    sim.initialize()
    statistics = {}
    for i in tqdm(range(num_steps)):
        action = "lockdown"
        do_nothing_pb = 0.03
        if np.random.random() < do_nothing_pb: action="lala"
        sim.execute(action)
        state = sim.get_statistics(kind='info')
        print(state)
        statistics[i] = state
        #print(state)

    plot_statistics(statistics)

