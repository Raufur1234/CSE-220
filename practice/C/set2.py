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


def time_reverse_signal(x : np.ndarray) -> np.ndarray:
    """
    Returns x[-n] for n = -8 to 8.
    Uses pure NumPy slicing (no loops).
    """
    # Step 1: Create an empty output array
    y = np.zeros_like(x)
    
    # Step 2: Reverse the array along the time axis
    # x[::-1] flips the array left-to-right.
    # However, we need to be careful about the center (n=0).
    # Since the array indices are: 0 -> n=-8, 8 -> n=0, 16 -> n=8
    # Reversing the array gives: index 0 -> original n=8, index 16 -> original n=-8
    # This perfectly implements x[-n] because the array is symmetric in length.
    y = x[::-1]
    
    return y
def time_reverse_signal_general(
        x: np.ndarray, 
        n_start: int, 
        n_end: int
    ) -> Tuple[np.ndarray, int, int]:
    """
    Generic time reversal: returns x[-n] for an arbitrary range [n_start, n_end].
    Unlike time_reverse_signal(), this handles asymmetric ranges correctly.
    
    The reversed signal lives on [-n_end, -n_start] — NOT the original range.
    Returns (y, out_start, out_end) so the caller knows the new index origin.
    
    Example: signal on [1, 4]  →  reversed signal on [-4, -1]
    """
    out_start = -n_end
    out_end   = -n_start
    y = np.zeros(out_end - out_start + 1)

    n_input        = np.arange(n_start, n_end + 1)
    input_indices  = n_input - n_start          # positions in x
    output_indices = -n_input - out_start       # positions in y  (= n_end - n_input)

    y[output_indices] = x[input_indices]

    return y, out_start, out_end

def odd_even_decomposition(x : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (odd_component, even_component) of the signal x[n].
    Formulas:
        even[n] = (x[n] + x[-n]) / 2
        odd[n]  = (x[n] - x[-n]) / 2
    """
    # Step 1: Get the time-reversed version of x
    x_rev = time_reverse_signal(x)
    
    # Step 2: Compute even and odd components using vectorized operations
    even_signal = (x + x_rev) / 2
    odd_signal  = (x - x_rev) / 2
    
    return odd_signal, even_signal




    
def main():
    img_root_path = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')
    reversed = time_reverse_signal(signal)
    plot(reversed, title='x[-n]', saveTo=f'{img_root_path}/x[-n].png')
    plot(time_reverse_signal(reversed), title='x[-(-n)]', saveTo=f'{img_root_path}/x[-(-n)].png')
    odd_signal, even_signal = odd_even_decomposition(signal)
    plot(odd_signal, title='Odd Signal(x[n])', saveTo=f'{img_root_path}/x_odd[n].png')
    plot(even_signal, title='Even Signal(x[n])', saveTo=f'{img_root_path}/x_even[n].png')


main()