import mpnum as mp


def sandwich_mpa(op, psi, mpa_type):
    """
        Calculates <op>_{psi} = tr(op*psi) for a state of specified mpa_type
    """
    if mpa_type == 'mps':
        psi = mp.mps_to_mpo(psi)
    elif mpa_type == 'pmps':
        psi = mp.pmps_to_mpo(psi)
    elif mpa_type == 'mpo':
        pass
    else:
        raise AssertionError('Unknown mpa_type')
    return mp.trace(mp.dot(op, psi))