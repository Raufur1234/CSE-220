import numpy as np
from transforms import DFTAnalyzer, FFTTransformer, NTTTransformer
x = np.random.randn(64) + 1j * np.random.randn(64)
d, f = DFTAnalyzer(), FFTTransformer()
assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9

# bonus 2: the NTT is exact, so it must agree with the float route to the digit
n = NTTTransformer()
k = np.random.randint(0, 100, 64)
assert np.array_equal(n.inverse(n.transform(k)), k)
pad = lambda v: np.concatenate([v, np.zeros(64, dtype=np.int64)])
a, b = np.random.randint(0, 100, 64), np.random.randint(0, 100, 64)
exact = n.inverse(n.transform(pad(a)) * n.transform(pad(b)))
approx = np.round(f.inverse(f.transform(pad(a)) * f.transform(pad(b))).real).astype(np.int64)
assert np.array_equal(exact, approx)
print("passed")
