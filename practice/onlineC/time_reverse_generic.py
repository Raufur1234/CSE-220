import numpy as np

def time_reverse(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns samples of x(-t) using interpolation.
    Works for ANY time array t (symmetric, asymmetric, non-uniform).
    """
    return np.interp(-t, t, x)