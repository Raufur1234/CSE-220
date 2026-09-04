import numpy as np
from transforms import DFTAnalyzer, FFTTransformer
x = np.random.randn(64) + 1j * np.random.randn(64)
d, f = DFTAnalyzer(), FFTTransformer()
assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
print("passed")