import numpy as np

def read_data(filename):
    """
    :param filename: File that we read data from
    :return: dataset and corresponding advantages
    """

    states_and_advs = np.genfromtxt(filename)
    states = states_and_advs[:, :5]
    advs = states_and_advs[:,5:]
    return states, advs

read_data('advantages.txt')

