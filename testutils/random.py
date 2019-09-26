"""
    Convenience functions to generate random hamiltonins, vectors, matrices, physical states and density matrices
"""

import numpy as np


def generate_random_hamiltonian(site_dims):
    """
        Generates a random Hamiltonian in the form of one local site operator and one local bond operator
        to be used for all sites and bonds
    :param site_dims: iterable of integers of local site dimension(s)
    :return: list of site operators as nparrays of dimension site_dim[i] x site_dim[i],
             List of bond operators, as nparrays of dimension site_dim^2 x site_dim^2
    """
    site_ops = []
    bond_ops = []
    for site_dim in site_dims:
        site_ops.append(np.random.rand(site_dim, site_dim))
        bond_ops.append(np.random.rand(site_dim**2, site_dim**2))
    site_ops = [site_op + site_op.T for site_op in site_ops]
    bond_ops = [bond_op + bond_op.T for bond_op in bond_ops]
    return site_ops, bond_ops


def generate_random_star_hamiltonian(site_dims, system_index):
    """
        Generates a random Hamiltonian in the form of one local site operator and one local bond operator for the
        sites to the left and to the right of the system
    :param site_dims: site_dims: iterable of integers of local site dimension(s)
    :param system_index: Index of the syste site in the chain
    :return: list of site operators as nparrays of dimension site_dim[i] x site_dim[i],
             List of bond operators, as nparrays of dimension site_dim^2 x site_dim^2
    """
    site_ops = []
    bond_ops = []
    for i in range(len(site_dims)):
        if i < system_index:
            site_ops.append(np.random.rand(site_dims[i], site_dims[i]))
            bond_ops.append(np.random.rand(site_dims[i] * site_dims[system_index],
                                           site_dims[i] * site_dims[system_index]))
        elif i == system_index:
            site_ops.append(np.random.rand(site_dims[system_index], site_dims[system_index]))
        else:
            site_ops.append(np.random.rand(site_dims[i], site_dims[i]))
            bond_ops.append(np.random.rand(site_dims[i] * site_dims[system_index],
                                           site_dims[i] * site_dims[system_index]))
    site_ops = [site_op + site_op.T for site_op in site_ops]
    bond_ops = [bond_op + bond_op.T for bond_op in bond_ops]
    return site_ops, bond_ops


def generate_random_matrices(site_dims):
    """
        Simplified version of generate_random_Hamiltonian, which takes an iterable of dimensions and returns
        quadratic matrices (nparrays) with shape: site_dim x site_dim
    :param site_dims: list of dimensions for which to generate random nparrays
    :return: list of random nparrays of shape site_dim x site_dim
    """
    site_ops = []
    for site_dim in site_dims:
        site_ops.append(np.random.rand(site_dim, site_dim))
    return site_ops


def generate_random_vetors(site_dims):
    """
        Takes an iterable of dimensions and returns vectors (nparrays) with shape: site_dim
    :param site_dims: list of dimensions for which to generate random nparrays
    :return: list of random nparrays of shape: site_dim
    """
    site_vecs = []
    for site_dim in site_dims:
        site_vecs.append(np.random.rand(site_dim))
    return site_vecs


def generate_random_density_matrices(site_dims):
    """
        Simplified version of generate_random_Hamiltonian, which takes an iterable of dimensions and returns
        density matrices (nparrays) with shape: site_dim x site_dim
    :param site_dims: list of dimensions for which to generate random density matrices
    :return: list of random nparrays as density matrices (positive, hermitian, trace 1 matrices)
    """
    site_ops = []
    for site_dim in site_dims:
        rand_mat = np.random.rand(site_dim, site_dim)
        pos_def_mat = rand_mat + rand_mat.T + np.eye(site_dim)
        site_ops.append(pos_def_mat/np.trace(pos_def_mat))
    return site_ops


def generate_random_state_vectors(site_dims):
    """
        Takes an iterable of dimensions and returns pyhsical states (nparrays) with shape: site_dim
    :param site_dims: list of dimensions for which to generate random physical states.
    :return: list of random nparrays as random physical states (vectors with norm 1)
    """
    site_vecs = []
    for site_dim in site_dims:
        rand_vec = np.random.rand(site_dim)
        site_vecs.append(rand_vec/np.linalg.norm(rand_vec))
    return site_vecs