import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_shift_signal(x: np.ndarray, k: int) -> np.ndarray:
    """
    Shifts the signal by k units. 
    If k > 0, shifts right. If k < 0, shifts left.
    """
    y = np.zeros_like(x)
    
    if k > 0:
        # Shift right: skip the first k elements in y, drop the last k in x
        y[k:] = x[:-k]
    elif k < 0:
        # Shift left: drop the last |k| elements in y, skip the first |k| in x
        y[:k] = x[-k:]
    else:
        # No shift
        y[:] = x
        
    return y

def time_scale_signal(x: np.ndarray, k: int,*args) -> np.ndarray:
    """
    Time scales the signal by a factor of k (where k is a positive integer).
    y[n] = x[k * n]
    """
    y = np.zeros_like(x)
    INF = 8
    
    # 1. Create an array of the actual time indices: [-8, -7, ..., 0, ..., 7, 8]
    n = np.arange(-INF, INF + 1)
    
    # 2. Calculate the corresponding time indices in the original signal
    orig_n = k * n
    
    # 3. Create a mask to only keep the indices that fall within our [-8, 8] window
    valid_mask = (orig_n >= -INF) & (orig_n <= INF)
    
    # 4. Map the valid time indices back to Python array indices (by adding INF)
    
    # y_indices is where we will write the data
    # x_indices is where we will read the data from
    y_indices = n[valid_mask] + INF
    x_indices = orig_n[valid_mask] + INF
    
    # 5. Copy the valid data over
    y[y_indices] = x[x_indices]
    
    return y


def main():
    img_root_path = 'images'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')

    plot(time_shift_signal(signal, 2), title='x[n-2]', saveTo=f'{img_root_path}/x[n-2].png')
    
    plot(time_shift_signal(signal, -2), title='x[n+2]', saveTo=f'{img_root_path}/x[n+2].png')
    
    plot(time_shift_signal(signal, 0), title='x[n+0]', saveTo=f'{img_root_path}/x[n+0].png')
    
    plot(time_scale_signal(signal, 2, True), title='x[2n]', saveTo=f'{img_root_path}/x[2n].png')
  
    plot(time_scale_signal(signal, 1, True), title='x[1n]', saveTo=f'{img_root_path}/x[1n].png')
    
        

main()

