"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:
"""


import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    if n<=1:
     return 1

    return 1 << (n - 1).bit_length()





class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        N=len(x)
        X= np.zeros(N,dtype=np.complex128)
        for k in range (0,N):
            for n in range (0,N):
                X[k] += x[n]*np.exp(-2j*np.pi*k*n/N)

        return X

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        N=len(spectrum)
        return np.conj(self.transform(np.conj(spectrum))) / N


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        N=len(x)
        if(N!=next_power_of_two(N)):
            raise ValueError
        x=np.array(x,dtype=np.complex128)
        width = N.bit_length()-1

        
        idx = np.arange(N)
        rev = np.zeros(N, dtype=np.intp)
        for b in range(width):
            rev |= ((idx >> b) & 1) << (width - 1 - b)
        x_bit_rev = x[rev]

        for s in range(1,int(np.log2(N)+1)):
            M= 2**s
            half = M//2
            # twiddles for this stage, computed once for the whole stage
            W = np.exp(-2j*np.pi*np.arange(half)/M)
           
            blocks = x_bit_rev.reshape(-1, M)
            g = blocks[:, :half]
            h = W * blocks[:, half:]
            top, bottom = g + h, g - h
            blocks[:, :half] = top
            blocks[:, half:] = bottom

        return x_bit_rev
    
    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        N = len(spectrum)
        return np.conj(self.transform(np.conj(spectrum))) / N



# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        if N <= 1:
            return x.copy()

        # Bluestein: k*n = (k^2 + n^2 - (k-n)^2) / 2, so X[k] = w[k] * (a * b)[k]
        # where a[n] = x[n]*w[n], b[m] = conj(w[m]) (b symmetric, support -(N-1)..N-1),
        # and (a*b) is a *linear* convolution -- evaluated via a radix-2 FFT of
        # length M = next_power_of_two(2N-1) so it doesn't need N itself to be a power of two.
        n = np.arange(N)
        w = np.exp(-1j * np.pi * n**2 / N)
        a = x * w

        M = next_power_of_two(2 * N - 1)
        a_pad = np.zeros(M, dtype=np.complex128)
        a_pad[:N] = a

        b = np.zeros(M, dtype=np.complex128)
        b[:N] = np.conj(w)
        b[M - N + 1:] = np.conj(w[1:][::-1])

        prod = super().transform(a_pad) * super().transform(b)
        conv = np.conj(super().transform(np.conj(prod))) / M
        return w * conv[:N]

    def inverse(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        if N <= 1:
            return spectrum.copy()
        return np.conj(self.transform(np.conj(spectrum))) / N
