import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
    # Initialize output array with zeros
    y = np.zeros_like(x)
    
    # Create an array representing the time indices n in the output
    n_out = np.arange(-INF, INF + 1)
    
    # Compute the corresponding n in the original signal
    n_in = n_out / k
    
    # Find indices where n_out is a multiple of k (n_in is an integer)
    is_multiple = (n_out % k == 0)
    
    # We only update y where n_out is a multiple of k
    # For those indices, y[INF + n_out] = x[INF + n_in]
    # n_in is an integer at these indices, so we can cast it to int
    y[INF + n_out[is_multiple]] = x[INF + n_in[is_multiple].astype(int)]
    
    return y

def time_scale_signal_interpolate(x : np.ndarray, k : int) -> np.ndarray:
    # Initialize output signal with zeros
    y = np.zeros_like(x)
    
    # Generate all output time indices n
    n_out = np.arange(-INF, INF + 1)
    
    # Compute corresponding original signal indices
    n_in = n_out / k
    
    # Handle exact multiples (same as Task 1)
    is_multiple = (n_out % k == 0)
    y[INF + n_out[is_multiple]] = x[INF + n_in[is_multiple].astype(int)]
    
    # Handle intermediate samples (where n_in is not an integer)
    is_interp = ~is_multiple  # complement mask
    if np.any(is_interp):
        # Get floor and ceil of n_in for interpolation points
        n_low = np.floor(n_in[is_interp]).astype(int)
        n_high = np.ceil(n_in[is_interp]).astype(int)
        
        # Average of the two original samples between which n_in lies
        y[INF + n_out[is_interp]] = (x[INF + n_low] + x[INF + n_high]) / 2
    
    return y

def time_scale_signal_weighted_interpolate(x : np.ndarray, k : int) -> np.ndarray:
    # Initialize output signal
    y = np.zeros_like(x)
    
    # Generate all output time indices
    n_out = np.arange(-INF, INF + 1)
    
    # Compute corresponding input indices
    n_in = n_out / k
    
    # Handle exact multiples (same as Task 1)
    is_multiple = (n_out % k == 0)
    y[INF + n_out[is_multiple]] = x[INF + n_in[is_multiple].astype(int)]
    
    # Handle intermediate samples
    is_interp = ~is_multiple
    if np.any(is_interp):
        # Get floor and ceil of n_in
        n_low = np.floor(n_in[is_interp]).astype(int)
        n_high = np.ceil(n_in[is_interp]).astype(int)
        
        # Calculate weights based on distance
        # n_in is floating point, so n_in - n_low gives fraction from low to high
        dist_low = n_in[is_interp] - n_low      # distance from lower neighbor
        dist_high = n_high - n_in[is_interp]    # distance from upper neighbor
        
        # Weighted interpolation: closer neighbor gets more weight
        y[INF + n_out[is_interp]] = (dist_high * x[INF + n_low] + 
                                     dist_low * x[INF + n_high])
    
    return y


def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

if __name__ == "__main__":
    main()
    
    
    
    def time_scale_general(x: np.ndarray, k) -> np.ndarray:
    """
    y[n] = x[k*n]
    k can be any positive/negative rational (int, float, or Fraction):
      k = 3      -> compression by 3   (like your original time_compress)
      k = 1/3    -> expansion by 3     (like your original time_scale_signal)
      k = 1      -> identity
      k = -1     -> time reversal
    """
    frac = Fraction(k).limit_denominator(1000)
    p, q = frac.numerator, frac.denominator

    y = np.zeros_like(x)
    n_out = np.arange(-INF, INF + 1)

    numer = p * n_out
    valid = (numer % q == 0)
    n_in = numer[valid] // q

    in_range = (n_in >= -INF) & (n_in <= INF)
    y[n_out[valid][in_range] + INF] = x[n_in[in_range] + INF]
    return y
