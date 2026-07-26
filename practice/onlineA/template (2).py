import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    y=np.zeros_like(t_query)
    x=x_original
    eps=1e-12
    for i,t in enumerate(t_query):
        pos=np.searchsorted(t_original,t)

        if pos>0 and abs(t_original[pos-1]-t)<eps:
            y[i]=x[pos-1]
        elif pos<len(t_original) and abs(t_original[pos]-t)<eps:
            y[i]=x[pos]
        else:
            if pos<=0:
                y[i]=x[0]
            elif pos>=len(t_original):
                y[i]=x[-1]
            else:
                y[i]=(x[pos-1]+x[pos])/2
            
    return y

def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    t_query=t/k
    x_new=interpolate_signal(t,x,t_query)
    return x_new


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    plt.figure()
    plt.plot(t,x,'b-',linewidth=4)
    plt.plot(t,y,'g-',linewidth=4)
    plt.title('BAL')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()
    
    

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate x at t_query, given known samples (t_original, x_original).
    Missing values = average of nearest left and right known samples.
    Exact matches return the known sample directly.
    """
    order = np.argsort(t_original)
    t_sorted = t_original[order]
    x_sorted = x_original[order]
    n = len(t_sorted)

    # Insertion index: t_sorted[idx-1] <= t_query < t_sorted[idx] (roughly)
    idx = np.searchsorted(t_sorted, t_query, side='left')
    idx_clipped = np.clip(idx, 0, n - 1)

    # Exact match: query lands precisely on a known sample
    exact_match = (idx < n) & np.isclose(t_sorted[idx_clipped], t_query)

    left_idx = np.clip(idx - 1, 0, n - 1)
    right_idx = np.clip(idx, 0, n - 1)

    left_val = x_sorted[left_idx]
    right_val = x_sorted[right_idx]

    averaged = 0.5 * (left_val + right_val)
    result = np.where(exact_match, x_sorted[idx_clipped], averaged)

    # Queries outside the known range are undefined -> NaN (dropped when plotting)
    out_of_range = (t_query < t_sorted[0]) | (t_query > t_sorted[-1])
    result = np.where(out_of_range, np.nan, result)

    return result