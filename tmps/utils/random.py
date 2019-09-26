import mpnum as mp
from tmps.utils.shape import get_shape_from_dims
import numpy as np


def get_random_mpa(mpa_type, dims, seed=102, rank=1):
    """
        Returns a random normalized mpa of specified type ('mps', 'mpo', 'pmps'), specified physical dimensions (dims),
        rank and numpy seed.
    """
    rng = np.random.RandomState(seed=seed)
    if mpa_type == 'mps':
        shape = get_shape_from_dims(dims)
        return mp.random_mpa(sites=len(shape), ldim=shape, rank=rank, randstate=rng, normalized=True,
                             force_rank=True)
    elif mpa_type == 'pmps':
        shape = get_shape_from_dims(dims, dims)
        return mp.random_mpa(sites=len(shape), ldim=shape, rank=rank, randstate=rng, normalized=True,
                             force_rank=True)
    elif mpa_type == 'mpo':
        shape = get_shape_from_dims(dims, dims)
        pmps = mp.random_mpa(sites=len(shape), ldim=shape, rank=rank, randstate=rng, normalized=True,
                             force_rank=True)
        return mp.pmps_to_mpo(pmps)
