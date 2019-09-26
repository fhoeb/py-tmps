"""
    Functions which allow to transform between mpnum shapes and chain shapes used for the embed/norm functions
"""


def get_shape_from_dim(L, axis_0, axis_1=None):
    """
        Constructs a chain shape tuple of length L with constant site dimensions everywhere
    :param L: Length of chain
    :param axis_0: Dimension of the first leg on each site (axis 0) as integer
    :param axis_1: Dimension of the second leg on each site (axis 1) as integer
    :return: chain shape tuple
    """
    if axis_1 is None:
        return tuple([(axis_0, )] * L)
    else:
        return tuple([(axis_0, axis_1)] * L)


def get_shape_from_dims(axis_0, axis_1=None):
    """
        Constructs a chain shape tuple from an array-like axis_0 and an optional array-like axis_1
    :param axis_0: Iterable of dimensions for the first leg on each site of the chain
    :param axis_1: Optional iterable of dimensions for the second leg on each site of the chain (of the same size
                   as axis_0)
    :return: chain shape tuple
    """
    if axis_1 is None:
        return tuple([(dim, )for dim in axis_0])
    else:
        assert len(axis_0) == len(axis_1)
        return tuple([(dim0, dim1) for dim0, dim1 in zip(axis_0, axis_1)])


def get_dims_from_shape(shape):
    """
        Transforms an mpnum shape into a chain tuple (only contains axis 0 dimensions of shape)
    :param shape: mpnum shape
    :return: tuple of axis 0 dimensions of shape
    """
    return tuple([dims[0] for dims in shape])


def get_shape_from_hamiltonian(h_site, axis_1=None):
    """
        Constructs a chain shape tuple from an iterable of site local operators and an optional array-like ancilla_dims
    :param h_site: Iterable of quadratic site local operators (is exhausted after this function call). Is used
                   to determine the axis 0 dimensions of the chain (for the first leg on each site)
    :param axis_1: Optional iterable of dimensions for the second leg on each site of the chain (of the same size
                   as axis_0)
    :return: chain shape tuple
    """
    shape = []
    if axis_1 is None:
        for site in h_site:
            assert site.shape[0] == site.shape[1]
            shape.append((site.shape[0], ))
    else:
        for site, dim1 in zip(h_site, axis_1):
            assert site.shape[0] == site.shape[1]
            shape.append((site.shape[0], dim1))
    return tuple(shape)


def check_shape(psi_0, mpa_type):
    """
        Checks if the state psi_0 has the correct form for
    :param psi_0: State whose mpa shape is checked
    :param mpa_type: MPA type to check for
    :return: Boolean. True, if psi_0 shape matches the required one for the selected mpa_type
    """
    len_state_shape = len(psi_0.shape[0])
    for site_shape in psi_0.shape:
        if len_state_shape != len(site_shape):
            print('Number of mpa legs does not agree on each site')
            return False
    if mpa_type == 'mps':
        if len_state_shape == 1:
            return True
        else:
            print('Number of physical legs on each site is not = 1')
            return False
    elif mpa_type == 'pmps' or mpa_type == 'mpo':
        if len_state_shape == 2:
            return True
        else:
            print('Number of physical legs on each site is not = 2')
            return False
    else:
        print('Unrecognized mpa_type')
        return False