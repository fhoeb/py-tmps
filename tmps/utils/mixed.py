import mpnum as mp


def get_maximally_mixed_pmps(dims, normalized=False):
    """
        Generates a maximally mixed state as pmps embedded on a chain with physical dimensions specified by dims
        The ancilla dimension are the same as the physical dimension on each site.
    :param dims: iterable of site local physical leg dimensions of the chain in which to embed
    :param normalized: Set True to return a normalized state
    :return: maximally mixed state as PMPS
    """
    mm_pmps = mp.eye(len(dims), dims)
    if not normalized:
        return mm_pmps
    else:
        return mm_pmps/mp.norm(mm_pmps)


def get_maximally_mixed_mpo(dims, normalized=False):
    """
        Generates a maximally mixed state as mpo embedded on a chain with physical dimensions specified by dims.
    :param dims: iterable of site local physical leg dimensions of the chain in which to embed
    :param normalized: Set True to return a normalized state
    :return: maximally mixed state as MPO
    """
    mm_mpo = mp.eye(len(dims), dims)
    if not normalized:
        return mm_mpo
    else:
        return mm_mpo/mp.trace(mm_mpo)


def get_maximally_mixed_state(mpa_type, dims, normalized=True):
    """
        Generates a maximally mixed state as mpo or pmps embedded on a chain with physical dimensions
        specified by site_dims.
    :param dims: iterable of site local physical leg dimensions of the chain in which to embed
    :param normalized: Set True to return a normalized state
    :return: maximally mixed state as MPO or PMPS
    """
    if mpa_type == 'mpo':
        return get_maximally_mixed_mpo(dims, normalized=normalized)
    elif mpa_type == 'pmps':
        return get_maximally_mixed_pmps(dims, normalized=normalized)
    else:
        raise AssertionError('Unsupported mpa_type')
