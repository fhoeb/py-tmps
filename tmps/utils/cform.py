def get_canonical_form(mpa):
    """
       Returns canonical form of mpa. Is either 'left', 'right' or None
    """
    cform = mpa.canonical_form
    if cform[0] == len(mpa) - 1:
        return 'left'
    elif cform[1] == 1:
        return 'right'
    else:
        return None


def canonicalize_to(mpa, to_cform='left'):
    """
        Canoonicalizes mpa to specified canonical form. NOP if to_cform is None or mpa is already in the desired
        canonical form
    """
    if to_cform is None:
        return
    cform = get_canonical_form(mpa)
    if cform is not None:
        if cform == to_cform:
            return
    if to_cform == 'right':
        mpa.canonicalize(right='afull')
    else:
        mpa.canonicalize(left='afull')


def canonicalize(mpa):
    """
        Left canonicalizes MPA, if its not already canonicalized and returns the canonical form of the mpa
        (String of either 'left' or 'right')
    """
    cform = get_canonical_form(mpa)
    if cform is None:
        mpa.canonicalize(left='afull')
        return 'left'
    else:
        return cform