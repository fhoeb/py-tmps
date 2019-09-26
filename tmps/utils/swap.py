import numpy as np
import mpnum as mp


def get_swap_tensor(dimL, dimR):
    """
        Returns a four legged tensor (dimR, dimL, dimL, dimR), which upon contracting the last two legs
        with a tensor with (dimL, dimR) swaps the sites to (dimR, dimL)
    """
    swap = np.zeros((dimR, dimL, dimL, dimR))
    for i in range(dimL):
        for j in range(dimR):
                swap[j, i, i, j] = 1
    return swap


def get_swap_gate(dimL, dimR):
    """
        Returns a Matrix dimR*dimL x dimL*dimR, which upon contracting taking the dot product
        with a vector dimL*dimR swaps the sites
    """
    swap = get_swap_tensor(dimL, dimR)
    return swap.reshape(dimL*dimR, dimL*dimR)


def get_swap_mpo(dimL, dimR):
    """
        Returns an mpo with four legs, which upon contracting by taking the dot product
        with an mps swaps two sites: (dimL; dimR) -> (dimR; dimL)
    """
    swap = get_swap_tensor(dimL, dimR)
    return mp.MPArray.from_array_global(swap, ndims=2)


def get_pmps_swap_tensor(dimL, dimR):
    """
        Returns tensor, which upon contraction swaps the tensor legs dimL and dimR:
        (dimL1, dimL2, dimR1, dimR2) -> (dimR1, dimR2, dimL1, dimL2)
        The returned swap tensor thus has dimensions:
        (dimR1, dimR2, dimL1, dimL2, dimL1, dimL2, dimR1, dimR2)
    :param dimL: Tuple of dimensions of the left legs of the tensor to swap (dimL1, dimL2)
    :param dimR: Tuple of dimensions of the right legs of the tensor to swap (dimR1, dimR2)
    :return: swap tensor
    """
    dimL1, dimL2 = dimL[0], dimL[1]
    dimR1, dimR2 = dimR[0], dimR[1]
    swap = np.zeros((dimR1, dimR2, dimL1, dimL2, dimL1, dimL2, dimR1, dimR2))
    for i in range(dimL1):
        for j in range(dimL2):
            for k in range(dimR1):
                for l in range(dimR2):
                    swap[k, l, i, j, i, j, k, l] = 1
    return swap


def get_pmps_swap_gate(dimL, dimR):
    """
        Returns matrix, which upon contraction swaps the tensor legs dimL and dimR
        The returned swap matrix thus has dimensions:
        (dimR1 * dimR2 * dimL1 * dimL2, dimL1 * dimL2 * dimR1 * dimR2)
    :param dimL: Tuple of dimensions of the left legs of the tensor to swap (dimL1, dimL2)
    :param dimR: Tuple of dimensions of the right legs of the tensor to swap (dimR1, dimR2)
    :return: swap matrix
    """
    dimL1, dimL2 = dimL[0], dimL[1]
    dimR1, dimR2 = dimR[0], dimR[1]
    swap = get_pmps_swap_tensor(dimL, dimR)
    return swap.reshape(dimL1*dimL2*dimR1*dimR2, dimL1*dimL2*dimR1*dimR2)


def get_pmps_swap_mpo(dimL, dimR):
    """
        Returns mpo eight legs, which upon contracting by taking the dot product with a pmps swaps
        two sites:
        (dimL1, dimL2; dimR1, dimR2) -> (dimR1, dimR2; dimL1, dimL2)
        (where the ones indexed by 2 are the ancilla legs of the respective sites)
    :param dimL: Tuple of dimensions of the left legs of the tensor to swap (dimL1, dimL2)
    :param dimR: Tuple of dimensions of the right legs of the tensor to swap (dimR1, dimR2)
    :return: swap mpo
    """
    swap = get_pmps_swap_tensor(dimL, dimR)
    return mp.MPArray.from_array_global(swap, ndims=2)