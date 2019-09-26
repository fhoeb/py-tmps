"""
    Functions for constructing propagation grids
"""
import numpy as np


def arange(start, tau, nof_steps):
    """
        Wrapper for numpy arange to avoid fp problems
    :param start: Start of the grid
    :param tau: Stepsize
    :param nof_steps: number of steps in the grid (= nof points - 1)
    :return: time grid as numpy array
    """
    return np.arange(0, nof_steps+1) * tau + start


def linspace(start, stop, nof_steps):
    """
        Wrapper for numpy linspace complementary to arange method
    :param start: Start of the grid
    :param stop: Endpoint of the grid
    :param nof_steps: number of steps in the grid (= nof points - 1)
    :return: time grid as numpy array, timestep
    """
    return np.linspace(start, stop, num=nof_steps+1, retstep=True)
