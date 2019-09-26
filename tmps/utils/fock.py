"""
    Contains generators for number state operators n, a and a_dag
"""
import numpy as np


def n(dim):
    return np.diag(np.arange(dim), k=0)


def a(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), k=1)


def a_dag(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), k=-1)
