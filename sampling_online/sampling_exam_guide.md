# Sampling Coding Test Guide

This is a copy-first reference for a NumPy/Matplotlib coding test on sampling. The companion file `sampling_skeleton.py` contains tested versions of the reusable helpers.

Start most programs with:

```python
import numpy as np
import matplotlib.pyplot as plt

from sampling_skeleton import *
```

## 1. Syllabus Boundary and How to Use This Guide

Included: the lecture from sampling terminology through the complete zero-order-hold (ZOH) frequency response, PDF pages 1-31.

- Uniform sampling and ambiguity
- Impulse-train sampling
- Multiplication in time / convolution in frequency
- Spectrum replication
- Nyquist-Shannon condition
- Oversampling, undersampling, and aliasing
- Ideal low-pass and sinc reconstruction
- Zero-order hold

Excluded: linear interpolation / first-order hold and the later DFT derivations. This guide uses `np.fft` only as a numerical spectrum-visualization tool, not as coverage of that later theory.

Sections 3 to 5 are the language reference: section 3 is plain Python (lists, tuples, dictionaries, sorting, comprehensions, files), section 4 is NumPy including `np.convolve` and `np.fft`, and section 5 is Matplotlib. Sections 6 onward are the sampling material itself. If you already know the language, skim the quick indexes at 4.15 and 5.9 and start at section 6.

Suggested exam workflow:

1. Write the signal formula and list every frequency in hertz.
2. Set `f_max = max(frequencies)` and check `fs > 2*f_max`.
3. Build a dense time array only for a smooth-looking reference curve.
4. Build the actual sample times from `fs`.
5. Evaluate the same vectorized signal function on both arrays.
6. Plot, label, and state the theoretical conclusion.
7. If reconstruction is requested, select ideal sinc or ZOH explicitly.

## 2. Formula and Notation Sheet

| Meaning | Hertz notation | Angular-frequency notation |
|---|---:|---:|
| Sampling frequency | `fs` samples/s | `omega_s = 2*pi*fs` rad/s |
| Sampling period | `T = 1/fs` s | `omega_s = 2*pi/T` |
| Highest signal frequency | `f_max` Hz | `omega_M = 2*pi*f_max` rad/s |
| Nyquist rate | `2*f_max` | `2*omega_M` |

The lecture uses the strict condition:

```text
fs > 2*f_max
omega_s > 2*omega_M
```

Sampling exactly at equality is fragile. For example, samples of a sine can all land on zero. Unless a question explicitly accepts the textbook shorthand, choose a rate strictly greater than the Nyquist rate.

Impulse-train model:

```text
p(t)  = sum_n delta(t - nT)
xp(t) = x(t)p(t) = sum_n x(nT) delta(t - nT)
```

Frequency-domain copies:

```text
P(j omega)  = (2*pi/T) sum_k delta(omega - k*omega_s)
Xp(j omega) = (1/T) sum_k X(j(omega - k*omega_s))
```

Each copy is spaced by `omega_s` and scaled by `1/T`. If adjacent copies overlap, aliasing has mixed them and an ideal filter cannot unmix them.

Ideal reconstruction:

```text
H(j omega) = T for |omega| < pi/T, otherwise 0
h(t)       = sinc(t/T)
xr(t)      = sum_n x[n] sinc((t - nT)/T)
```

NumPy uses the normalized sinc:

```text
np.sinc(u) = sin(pi*u)/(pi*u)
```

Zero-order hold:

```text
h0(t) = 1 on [0, T), otherwise 0
H0(j omega) = T exp(-j*omega*T/2) sinc(omega*T/(2*pi))
H0(f)       = T exp(-j*pi*f*T) sinc(f*T)
```

The magnitude is `T*abs(sinc(f*T))`. It has DC gain `T`, droops with frequency, and has its first null at `f = fs = 1/T`.

## 3. Python Essentials

Everything in this section is plain Python, no NumPy. In a timed test the marks are usually lost here — a wrong dictionary lookup, a list that was aliased instead of copied, a sort that sorted the wrong field — rather than in the signal processing. Read this before section 4.

### 3.1 Numbers, operators, and printing

```python
a = 7
b = 2
print(a + b, a - b, a * b)   # 9 5 14
print(a / b)                 # 3.5   true division, always float
print(a // b)                # 3     floor division, drops the remainder
print(a % b)                 # 1     remainder
print(a ** b)                # 49    power  (NOT a^b -- ^ is bitwise xor)
print(abs(-3), round(3.14159, 2), min(4, 9), max(4, 9), sum([1, 2, 3]))
```

Gotchas that cost marks:

- `1/3` is `0.333...` but `1//3` is `0`.
- `0.1 + 0.2 == 0.3` is `False`. Compare floats with a tolerance: `abs(x - y) < 1e-9`, or `math.isclose(x, y)`, or NumPy's `np.isclose`.
- `^` is exclusive-or, not exponentiation. `2 ** 10` is 1024; `2 ^ 10` is 8.
- `int(3.9)` is `3`: it truncates toward zero, it does not round.

Type conversion:

```python
int("12")      # 12      string to int
float("2.5")   # 2.5
str(3.5)       # "3.5"
int(3.9)       # 3       truncates toward zero
round(3.9)     # 4
bool(0), bool(""), bool([])   # all False -- "empty is falsy"
```

### 3.2 Strings and f-strings

```python
name = "fs"
value = 44100.0

print(f"{name} = {value}")         # fs = 44100.0
print(f"{value:.2f}")              # 44100.00      2 decimals
print(f"{value:.3e}")              # 4.410e+04     scientific
print(f"{value:10.2f}")            # width 10, right aligned
print(f"{name:>8}|{name:<8}|{name:^8}")   # right, left, centre padded
print(f"{0.1234:.1%}")             # 12.3%
print(f"{255:04d}")                # 0255          zero-padded integer
print(f"{value = }")               # value = 44100.0  (prints the expression too)
```

Useful string methods:

```python
s = "  10, 20 , 30  "
s.strip()                              # "10, 20 , 30"   removes outer whitespace
s.split(",")                           # ["  10", " 20 ", " 30  "]
[p.strip() for p in s.split(",")]      # ["10", "20", "30"]
"-".join(["a", "b", "c"])              # "a-b-c"
"Hello".lower(), "Hello".upper()
"3.5 Hz".replace("Hz", "hertz")
"cos(2*pi*5*t)".startswith("cos")      # True
"5" in "cos(2*pi*5*t)"                 # True   substring test
len("abc")                             # 3
```

Multi-line report text:

```python
report = (
    f"fs       = {fs:.1f} Hz\n"
    f"f_max    = {f_max:.1f} Hz\n"
    f"aliasing = {'yes' if fs <= 2 * f_max else 'no'}"
)
print(report)
```

### 3.3 Lists

A list is an ordered, mutable sequence. It is the default container for "a bunch of values whose order matters".

```python
frequencies = [5.0, 12.0, 30.0]

frequencies[0]        # 5.0            first
frequencies[-1]       # 30.0           last
frequencies[1:3]      # [12.0, 30.0]   slice, stop is exclusive
frequencies[::-1]     # reversed copy
len(frequencies)      # 3

frequencies.append(45.0)          # add ONE item at the end
frequencies.extend([50.0, 60.0])  # add SEVERAL items
frequencies.insert(0, 1.0)        # insert at an index
frequencies.remove(30.0)          # remove the first matching VALUE
last = frequencies.pop()          # remove and return the last item
first = frequencies.pop(0)        # remove and return index 0
frequencies.index(12.0)           # position of a value
frequencies.count(12.0)           # how many times it occurs
frequencies.sort()                # sort IN PLACE, returns None
frequencies.reverse()             # reverse IN PLACE
frequencies.clear()               # empty it
```

`append` versus `extend` is a classic slip:

```python
a = [1, 2]
a.append([3, 4])   # [1, 2, [3, 4]]   one new item that happens to be a list
b = [1, 2]
b.extend([3, 4])   # [1, 2, 3, 4]     four items
```

Copying. Assignment does **not** copy; both names point at the same list:

```python
a = [1, 2, 3]
b = a
b[0] = 99
print(a)            # [99, 2, 3]  -- a changed too

c = a.copy()        # or list(a), or a[:]
c[0] = 0
print(a[0])         # 99 -- unaffected
```

Nested lists need a deep copy, otherwise the inner lists are still shared:

```python
import copy

grid = [[0, 0], [0, 0]]
shallow = grid.copy()
shallow[0][0] = 5
print(grid[0][0])         # 5   -- inner lists were shared

deep = copy.deepcopy(grid)
deep[0][0] = 9
print(grid[0][0])         # 5   -- unaffected
```

Never build a grid with `[[0] * 2] * 2`: that repeats the *same* inner list twice. Use a comprehension: `[[0] * 2 for _ in range(2)]`.

### 3.4 Tuples

A tuple is an ordered, **immutable** sequence. Use it for a fixed-size record and for returning several values at once.

```python
point = (0.0, 1.0)
x, y = point            # unpacking
single = (5,)           # a one-element tuple NEEDS the trailing comma
not_a_tuple = (5)       # this is just the integer 5
```

Functions return tuples all the time; the skeleton does:

```python
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
fig, ax = plt.subplots()
f, amplitude = magnitude_spectrum(x, fs)
grid, stack, total = spectrum_replicas(f, amplitude, fs)
```

Swapping and starred unpacking:

```python
a, b = 1, 2
a, b = b, a                      # swap, no temporary needed

first, *rest = [1, 2, 3, 4]      # first=1, rest=[2, 3, 4]
*head, last = [1, 2, 3, 4]       # head=[1, 2, 3], last=4
```

Tuples are hashable, so they can be dictionary keys; lists cannot:

```python
cache = {(5.0, 100.0): "ok"}     # (frequency, fs) -> verdict
cache[(5.0, 100.0)]
```

### 3.5 Dictionaries

A dictionary maps keys to values. Use it for named parameters, a lookup table, or per-case results.

```python
config = {"fs": 100.0, "duration": 1.0, "f_max": 12.0}

config["fs"]                   # 100.0    KeyError if the key is missing
config.get("phase")            # None     safe: no exception
config.get("phase", 0.0)       # 0.0      with a default

config["phase"] = np.pi / 4    # add or overwrite
config.update({"amplitude": 2.0, "fs": 200.0})   # bulk add/overwrite
config.setdefault("phase", 0.0)    # insert only if missing, return the value

"fs" in config                 # True     membership tests KEYS
del config["phase"]
value = config.pop("amplitude")           # remove and return
value = config.pop("missing", None)       # with a default, no exception
len(config)
```

Iterating:

```python
for key in config:                  # keys
    print(key)

for key, value in config.items():   # key/value pairs -- the usual form
    print(f"{key}: {value}")

list(config.keys())
list(config.values())
```

A dict keeps insertion order, so `items()` comes back in the order you inserted.

Counting and grouping without extra imports:

```python
verdicts = ["alias", "ok", "alias", "ok", "alias"]

counts = {}
for v in verdicts:
    counts[v] = counts.get(v, 0) + 1
# {"alias": 3, "ok": 2}

groups = {}
for name, f in [("a", 5.0), ("b", 30.0), ("c", 8.0)]:
    key = "low" if f < 10 else "high"
    groups.setdefault(key, []).append(name)
# {"low": ["a", "c"], "high": ["b"]}
```

Dict comprehension, and building one from two lists:

```python
tones = {f"tone_{i}": f for i, f in enumerate([5.0, 12.0, 30.0])}
nyquist = {f: 2 * f for f in [5.0, 12.0, 30.0]}
paired = dict(zip(["a", "b"], [1, 2]))       # {"a": 1, "b": 2}
```

A dict of results is the cleanest way to answer a multi-part question:

```python
results = {}
for fs in [10.0, 25.0, 50.0]:
    results[fs] = {"T": 1 / fs, "aliased": fs <= 2 * f_max}

for fs, info in results.items():
    print(f"fs={fs:6.1f}  T={info['T']:.4f}  aliased={info['aliased']}")
```

### 3.6 Sets

A set is an unordered collection of unique, hashable items. Use it for "which distinct frequencies appear" and for fast membership tests.

```python
frequencies = {5.0, 12.0, 5.0, 30.0}    # {5.0, 12.0, 30.0} -- duplicate dropped
unique = set([5.0, 12.0, 5.0])          # from a list
frequencies.add(7.0)
frequencies.discard(99.0)               # no error if absent (remove() raises)

{1, 2, 3} | {3, 4}      # union          {1, 2, 3, 4}
{1, 2, 3} & {3, 4}      # intersection   {3}
{1, 2, 3} - {3, 4}      # difference     {1, 2}
5.0 in frequencies      # True
sorted(frequencies)     # back to an ordered LIST
```

`set()` makes an empty set; `{}` makes an empty **dict**.

### 3.7 Sorting

`sorted(iterable)` returns a new list. `list.sort()` sorts in place and returns `None` — `x = mylist.sort()` giving `None` is a very common bug.

```python
values = [3.0, 1.0, 2.0]

ascending = sorted(values)                 # [1.0, 2.0, 3.0]   values unchanged
descending = sorted(values, reverse=True)
values.sort()                              # in place; returns None
```

Sorting by a computed key with `key=`:

```python
tones = [("a", 12.0), ("b", 5.0), ("c", 30.0)]

by_frequency = sorted(tones, key=lambda item: item[1])
# [("b", 5.0), ("a", 12.0), ("c", 30.0)]

loudest_first = sorted(tones, key=lambda item: item[1], reverse=True)
names_by_length = sorted(["alpha", "be", "cee"], key=len)
```

Sorting a dictionary — pick the field you want:

```python
peaks = {"5 Hz": 1.0, "12 Hz": 0.4, "30 Hz": 2.5}

sorted(peaks)                                   # keys, ascending
sorted(peaks.items(), key=lambda kv: kv[1])     # by value, ascending
sorted(peaks.items(), key=lambda kv: -kv[1])    # by value, descending
dict(sorted(peaks.items(), key=lambda kv: kv[1], reverse=True))   # back to a dict
max(peaks, key=peaks.get)                       # "30 Hz" -- key of the largest value
```

Multi-level sort: return a tuple from the key. Tuples compare left to right.

```python
records = [("b", 5.0), ("a", 5.0), ("c", 1.0)]
sorted(records, key=lambda r: (r[1], r[0]))     # by frequency, ties broken by name
```

`sorted` is stable: items that compare equal keep their original relative order, so you can sort by the secondary key first and then by the primary key.

Related built-ins:

```python
min(tones, key=lambda item: item[1])
max(tones, key=lambda item: item[1])
sum(f for _, f in tones)
any(f > 20 for _, f in tones)     # True if at least one passes
all(f > 0 for _, f in tones)      # True if every one passes
```

For numeric arrays prefer NumPy's `np.sort` / `np.argsort` (section 4.7); they are vectorized and keep everything as arrays.

### 3.8 Loops, `range`, `enumerate`, `zip`

```python
for i in range(5):            # 0 1 2 3 4       stop is exclusive
    ...
for i in range(1, 6):         # 1 2 3 4 5
    ...
for i in range(0, 10, 2):     # 0 2 4 6 8       step
    ...
for i in range(5, 0, -1):     # 5 4 3 2 1       countdown
    ...
```

`enumerate` gives index and item together; `zip` walks several sequences in lockstep and stops at the shortest:

```python
for n, value in enumerate(x_samples):
    print(f"x[{n}] = {value:.3f}")

for n, value in enumerate(x_samples, start=1):   # 1-based labels
    ...

for f, amplitude in zip(frequencies, amplitudes):
    print(f, amplitude)

for n, (t_n, x_n) in enumerate(zip(t_samples, x_samples)):
    ...
```

Loop control:

```python
for f in frequencies:
    if f <= 0:
        continue          # skip this item
    if f > fs / 2:
        break             # leave the loop entirely
else:
    print("loop finished without break")   # for/else runs only if no break

while fs <= 2 * f_max:
    fs *= 2               # keep doubling until Nyquist is satisfied
```

Never modify a list while iterating over it — build a new list instead.

### 3.9 Comprehensions

A comprehension is a loop that builds a container in one expression.

```python
squares = [n ** 2 for n in range(5)]                   # list
evens = [n for n in range(10) if n % 2 == 0]           # with a filter
labels = [f"{f:.0f} Hz" for f in frequencies]
folded = [f if f <= fs / 2 else fs - f for f in frequencies]   # if/else goes FIRST

pairs = [(i, j) for i in range(2) for j in range(2)]   # nested loops
grid = [[0.0] * 3 for _ in range(2)]                   # correct 2x3 of zeros

unique_rates = {2 * f for f in frequencies}            # set comprehension
nyquist = {f: 2 * f for f in frequencies}              # dict comprehension
total = sum(f ** 2 for f in frequencies)               # generator, no list built
```

Note the two positions of `if`: a trailing `if` filters items out; an `if ... else` before the `for` chooses what value to emit and keeps every item.

For numeric work, prefer a vectorized NumPy expression over a comprehension that loops over thousands of time instants (section 4.5).

### 3.10 Functions

```python
def nyquist_rate(f_max):
    """Return twice the highest frequency, in hertz."""
    return 2.0 * f_max


def describe(fs, f_max, strict=True):          # strict has a DEFAULT value
    rate = nyquist_rate(f_max)
    ok = fs > rate if strict else fs >= rate
    return ok, rate                            # returns a tuple


ok, rate = describe(100.0, 12.0)               # positional
ok, rate = describe(fs=100.0, f_max=12.0)      # keyword -- clearer, order free
ok, rate = describe(100.0, 12.0, strict=False)
```

Rules worth remembering:

- Parameters with defaults must come after parameters without.
- **Never** use a mutable default (`def f(items=[])`); the same list is reused across calls. Write `def f(items=None): items = [] if items is None else items`.
- A function without an explicit `return` returns `None`.
- Arguments are passed by reference: mutating a list or dict inside a function is visible to the caller; rebinding the name is not.

Flexible argument lists:

```python
def total(*values):            # collects positional arguments into a TUPLE
    return sum(values)

total(1, 2, 3)                 # 6


def plot_it(t, x, **style):    # collects keyword arguments into a DICT
    plt.plot(t, x, **style)    # and ** unpacks a dict back into keywords


plot_it(t, x, color="C1", linewidth=2)

args = [t, x]
kwargs = {"color": "C1"}
plot_it(*args, **kwargs)       # * unpacks a list into positional arguments
```

`lambda` is a one-expression anonymous function, used mainly as a `key=` or as a quick signal definition:

```python
signal = lambda t: np.sin(2 * np.pi * 5 * t)
sorted(tones, key=lambda item: item[1])
```

For anything longer than one expression write a `def`. A signal function used with the skeleton must be *vectorized*: it must accept an array of times and return an array of the same shape, which happens automatically if the body uses only NumPy operations.

### 3.11 Conditionals and truthiness

```python
if fs > 2 * f_max:
    verdict = "no aliasing"
elif fs == 2 * f_max:
    verdict = "critical, fragile"
else:
    verdict = "aliasing"

verdict = "ok" if fs > 2 * f_max else "aliased"    # conditional expression
```

Chained comparisons work as written in mathematics:

```python
if 0 < f < fs / 2:
    ...
```

Falsy values: `0`, `0.0`, `""`, `[]`, `{}`, `()`, `set()`, `None`. Everything else is truthy. Test emptiness with `if not items:`.

`==` compares values, `is` compares identity — use `is` only with `None`, `True` and `False`.

On NumPy arrays, `and` / `or` / `not` raise `ValueError: truth value of an array ... is ambiguous`. Use `&`, `|`, `~` with parentheses, or `np.any` / `np.all` (section 4.4).

### 3.12 Errors, exceptions, and assertions

```python
try:
    fs = float(text)
    T = 1.0 / fs
except ValueError:
    print("not a number")
except ZeroDivisionError:
    print("fs must not be zero")
else:
    print("parsed fine")          # runs only when nothing was raised
finally:
    print("always runs")
```

Raising your own, the way the skeleton validates its inputs:

```python
if fs <= 0:
    raise ValueError("fs must be positive")
```

`assert` states an invariant and doubles as documentation:

```python
assert t.shape == x.shape, "time and signal arrays must match"
```

Exceptions you will actually hit in this test: `IndexError` (index past the end), `KeyError` (missing dict key), `ValueError` (bad conversion or shape mismatch), `TypeError` (`math.sin` on an array), `ZeroDivisionError`, `NameError` (typo), `AttributeError` (`.append` on a tuple or on a NumPy array).

### 3.13 Input, output, and files

```python
fs = float(input("sampling rate in Hz: "))
n = int(input("number of samples: "))

line = input()                                    # "5 12 30"
values = [float(part) for part in line.split()]   # [5.0, 12.0, 30.0]
```

Command-line arguments:

```python
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "inputs/1.txt"
```

Files — always with `with`, which closes the file even if an error is raised:

```python
with open("inputs/1.txt") as handle:
    text = handle.read()              # whole file as one string

with open("inputs/1.txt") as handle:
    lines = [line.strip() for line in handle if line.strip()]

with open("report.txt", "w") as handle:            # "w" truncates, "a" appends
    handle.write(f"fs = {fs}\n")
    handle.write("\n".join(labels))
```

Parsing a typical test input file (first line `fs`, second line the frequencies):

```python
with open(path) as handle:
    lines = [line.strip() for line in handle if line.strip()]

fs = float(lines[0])
frequencies = [float(p) for p in lines[1].split()]
```

For numeric arrays, `np.loadtxt` / `np.savetxt` do this in one line (section 4.13).

### 3.14 The `math` module, and when not to use it

```python
import math

math.pi, math.e
math.sin(math.pi / 4), math.cos(0.0), math.exp(1.0)
math.sqrt(2.0), math.log(math.e), math.log10(100.0)
math.floor(3.7), math.ceil(3.2), math.fabs(-2.0)
math.isclose(0.1 + 0.2, 0.3)       # True -- the right way to compare floats
math.inf, math.nan, math.isnan(math.nan)
math.gcd(12, 18), math.factorial(5)
math.degrees(math.pi), math.radians(180.0)
```

`math` functions take **one scalar** and return one scalar. Passing an array raises `TypeError: only length-1 arrays can be converted`. For arrays use the NumPy function of the same name: `np.sin`, `np.exp`, `np.sqrt`, `np.pi`. Inside a vectorized signal function, always use `np`.

### 3.15 Modules, `__main__`, and small classes

```python
import numpy as np                       # standard alias
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, plot_sampling   # selected names
from sampling_skeleton import *          # everything in its __all__


def main():
    ...


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__":` guard lets the same file be imported by a grader without running your plotting code.

A small class can bundle a signal's parameters, though a dict or a tuple is usually enough here:

```python
class Tone:
    def __init__(self, frequency_hz, amplitude=1.0, phase=0.0):
        self.frequency_hz = frequency_hz
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, t):               # makes the instance callable
        return self.amplitude * np.cos(
            2 * np.pi * self.frequency_hz * t + self.phase
        )


tone = Tone(5.0, amplitude=2.0)
t_samples, x_samples = sample_signal(tone, fs=50.0, start=0.0, stop=1.0)
```

### 3.16 Python list versus NumPy array

| Operation | Python list | NumPy array |
|---|---|---|
| `a + b` | concatenation | elementwise addition |
| `a * 2` | repeats the list | doubles every element |
| `a - b` | `TypeError` | elementwise subtraction |
| elementwise math | needs a loop or comprehension | built in, vectorized |
| mixed types | allowed | one `dtype` for the whole array |
| `len(a)` | number of items | length of the first axis; use `a.size`, `a.shape` |
| `a[:]` | a copy | a **view** — use `.copy()` |
| speed | slower, boxed objects | contiguous, fast |

Convert with `np.asarray(list_of_numbers)` on the way in and `array.tolist()` on the way out. Do the arithmetic in NumPy.

## 4. NumPy Essentials

Every function below is listed with its signature, what it returns, the gotcha that usually bites, and where it is used in a sampling problem. Section 4.15 is a one-screen index of the whole lot.

### 4.1 Array creation

| Call | What you get |
|---|---|
| `np.array(obj, dtype=None)` | array copied from a list/tuple/nested list |
| `np.asarray(obj)` | same, but no copy if it is already an array |
| `np.arange(start, stop, step)` | values `start, start+step, ...`, **stop excluded** |
| `np.linspace(start, stop, num)` | `num` points, **both endpoints included** |
| `np.zeros(shape)` / `np.ones(shape)` | filled with 0 / 1 |
| `np.full(shape, value)` | filled with `value` |
| `np.empty(shape)` | uninitialised garbage — fill it before reading |
| `np.zeros_like(a)` / `np.ones_like(a)` / `np.full_like(a, v)` | same shape and dtype as `a` |
| `np.eye(n)` / `np.identity(n)` | identity matrix |
| `np.fromiter(gen, dtype, count=-1)` | array from a generator |

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
integers = np.arange(0, 8, 2)         # [0, 2, 4, 6]  -- 8 is excluded
grid = np.linspace(0.0, 1.0, 5)       # [0, 0.25, 0.5, 0.75, 1.0]
z = np.zeros(4)
o = np.ones(4)
c = np.full(4, 7.0)
same_shape = np.zeros_like(a)
```

Sampling-related forms:

```python
fs = 20.0
T = 1.0 / fs
t_samples = np.arange(0.0, 1.0, T)       # half-open [0, 1)
t_dense = np.linspace(0.0, 1.0, 2001)    # reference plot, includes 1.0

# More reliable when duration*fs should be an integer:
N = int(1.0 * fs)
t_samples = np.arange(N) / fs
```

`np.arange` with a **float** step accumulates rounding error and can emit one extra or one missing point; `np.arange(N) / fs` never does, because the division happens once per exact integer. Use `np.arange` for a known step, `np.linspace(start, stop, count)` for a known number of points, and never confuse `linspace`'s third argument with a sample rate.

`np.linspace` extras:

```python
np.linspace(0, 1, 5, endpoint=False)        # [0, 0.2, 0.4, 0.6, 0.8]  like arange
t, step = np.linspace(0, 1, 5, retstep=True)   # also returns the spacing
```

### 4.2 Inspecting, reshaping, and copying

| Call | Meaning |
|---|---|
| `a.shape` | tuple of lengths per axis, e.g. `(100,)` or `(3, 100)` |
| `a.ndim` | number of axes |
| `a.size` | total number of elements |
| `a.dtype` | element type (`float64`, `complex128`, `int64`, `bool`) |
| `a.astype(t)` | converted **copy** |
| `a.copy()` | independent copy |
| `a.reshape(shape)` | same data, new shape; one axis may be `-1` ("infer") |
| `a.ravel()` / `a.flatten()` | flatten to 1-D; `ravel` returns a view when it can, `flatten` always copies |
| `a.T` / `np.transpose(a)` | swap axes |
| `a[:, None]`, `np.newaxis`, `np.expand_dims(a, 1)` | insert a length-1 axis |
| `np.squeeze(a)` | drop every length-1 axis |
| `a.tolist()` | back to nested Python lists |

```python
x = np.array([1, 2, 3], dtype=np.int64)
print(x.shape)   # (3,)
print(x.ndim)    # 1
print(x.size)    # 3
print(x.dtype)   # int64

y = x.copy()               # independent data
y = x.astype(float)        # converted copy
row = x.reshape(1, -1)     # shape (1, 3)
column = x.reshape(-1, 1)  # shape (3, 1)
```

Reshaping rules: the product of the new shape must equal `a.size`, or NumPy raises `ValueError: cannot reshape array of size 6 into shape (4,)`. Exactly one axis may be `-1`, and NumPy solves for it. Reshape fills in C order (last axis varies fastest) unless you pass `order="F"`.

```python
x = np.arange(6)                 # [0 1 2 3 4 5]
x.reshape(2, 3)                  # [[0 1 2], [3 4 5]]
x.reshape(3, 2)                  # [[0 1], [2 3], [4 5]]
x.reshape(-1, 2)                 # (3, 2), rows inferred
x.reshape(2, 3).T                # (3, 2), transposed -- NOT the same values
x.reshape(2, 3).ravel()          # back to [0 1 2 3 4 5]
```

Reshaping a sampled record into frames is the one place it really comes up:

```python
x = np.arange(12.0)
frames = x.reshape(-1, 4)        # 3 frames of 4 samples each
frame_means = frames.mean(axis=1)   # one value per frame
```

`reshape` returns a **view** when it can: writing to the reshaped array writes to the original. Call `.copy()` if you need independence.

Sampling sanity check:

```python
t = np.arange(8) / 8.0
x = np.sin(2 * np.pi * 1.0 * t)
assert t.shape == x.shape
assert t.ndim == 1
```

### 4.3 Indexing, slicing, and assignment

```python
x = np.array([10, 20, 30, 40, 50])
first = x[0]
last = x[-1]
middle = x[1:4]       # indices 1, 2, 3
every_other = x[::2]
reversed_x = x[::-1]
x[1:3] = 0            # assign into a slice
```

Fancy indexing with a list or array of positions returns a **copy**:

```python
picked = x[[0, 2, 4]]                  # [10, 30, 50]
order = np.array([2, 0, 1])
x[:3][order]                           # reordered copy
```

Two-dimensional indexing:

```python
m = np.arange(12).reshape(3, 4)
m[1, 2]        # single element, row 1 column 2
m[1]           # whole row 1, shape (4,)
m[:, 2]        # whole column 2, shape (3,)
m[0:2, 1:3]    # sub-block
```

Sampling-related slicing:

```python
fs = 100
t = np.arange(100) / fs
x = np.cos(2 * np.pi * 5 * t)
first_half = x[:50]
even_numbered_samples = x[::2]         # this is DECIMATION by 2: fs becomes fs/2
```

Basic slicing returns a **view**. Use `.copy()` before modifying a slice when the original must stay unchanged.

### 4.4 Masks, Boolean operators, and `np.where`

```python
x = np.array([-2, -1, 0, 1, 2])
positive = x > 0            # array([False, False, False, True, True])
selected = x[positive]      # [1, 2]        boolean indexing returns a COPY
x_clipped = x.copy()
x_clipped[x_clipped < 0] = 0

inside = (x >= -1) & (x <= 1)
outside = (x < -1) | (x > 1)
not_inside = ~inside
y = np.where(x >= 0, x, -x)        # condition, value-if-true, value-if-false
```

Use `&`, `|`, `~` for array logic, not Python's `and`, `or`, `not`, and put each comparison in parentheses — `&` binds tighter than `>=`.

Reductions over masks:

```python
np.any(x > 1)               # True if at least one element passes
np.all(x > -5)              # True if every element passes
np.count_nonzero(x > 0)     # how many pass
np.nonzero(x > 0)           # tuple of index arrays of the passing positions
np.flatnonzero(x > 0)       # the same indices as a flat array
np.isfinite(x).all()        # no nan, no inf
np.isnan(x), np.isinf(x)
```

`np.where` has a second personality: called with one argument it returns the indices of the true entries, the same as `np.nonzero`.

```python
indices = np.where(x > 0)[0]     # [3, 4]
```

Piecewise signal:

```python
t = np.linspace(-2.0, 2.0, 1001)
x = np.zeros_like(t)
mask_rise = (t >= -1.0) & (t < 0.0)
mask_fall = (t >= 0.0) & (t <= 1.0)
x[mask_rise] = t[mask_rise] + 1.0
x[mask_fall] = 1.0 - t[mask_fall]

# Same triangle with np.where:
x2 = np.where(np.abs(t) <= 1.0, 1.0 - np.abs(t), 0.0)
assert np.allclose(x, x2)
```

`np.select` handles more than two branches, and `np.piecewise` takes functions:

```python
x3 = np.select([t < -1, mask_rise, mask_fall], [0.0, t + 1.0, 1.0 - t], default=0.0)
x4 = np.piecewise(t, [mask_rise, mask_fall],
                  [lambda z: z + 1.0, lambda z: 1.0 - z, 0.0])
```

In `np.select` and `np.where` every branch expression is evaluated for the whole array, so guard anything that could divide by zero.

### 4.5 Broadcasting and vectorization

Broadcasting: NumPy compares shapes axis by axis from the right. Two axes are compatible if they are equal or one of them is 1; a length-1 axis is stretched.

```python
column = np.array([1, 2, 3])[:, None]   # (3, 1)
row = np.array([10, 20])[None, :]       # (1, 2)
table = column + row                    # (3, 2)
```

```text
(3, 1)
(1, 2)
------
(3, 2)
```

Sampling-related vectorization:

```python
frequencies = np.array([3.0, 7.0, 12.0])
t = np.arange(100) / 100.0

# One row per tone; shape is (3, 100).
tones = np.sin(2 * np.pi * frequencies[:, None] * t[None, :])
x = tones.sum(axis=0)                   # sum the tones -> shape (100,)
```

This same trick builds the sinc reconstruction matrix in one line (section 9). `axis=0` collapses rows, `axis=1` collapses columns, and omitting `axis` collapses everything to a scalar.

If a formula truly cannot be vectorized, `np.vectorize(f)` wraps a scalar function so it accepts arrays — but it is a convenience loop, not a speed-up.

### 4.6 Combining, padding, differences, and reductions

| Call | Meaning |
|---|---|
| `np.concatenate([a, b])` | join along an existing axis |
| `np.stack([a, b], axis=0)` | join along a **new** axis |
| `np.vstack` / `np.hstack` | stack as rows / columns |
| `np.pad(a, (left, right), constant_values=0)` | add values at the ends |
| `np.tile(a, reps)` | repeat the whole array `reps` times |
| `np.repeat(a, n)` | repeat **each element** `n` times |
| `np.roll(a, k)` | circular shift by `k` |
| `np.flip(a)` | reverse |
| `np.append(a, v)` | copy with items added — avoid in loops, it reallocates |
| `np.insert` / `np.delete` | copy with items inserted / removed |
| `np.diff(a)` | successive differences, length `n-1` |
| `np.cumsum(a)` / `np.cumprod(a)` | running sum / product |
| `np.sum`, `np.mean`, `np.std`, `np.var`, `np.min`, `np.max`, `np.ptp` | reductions, all take `axis=` |
| `np.argmin`, `np.argmax` | **index** of the extreme value |
| `np.allclose(a, b)`, `np.isclose(a, b)` | float comparison with tolerance |
| `np.array_equal(a, b)` | exact equality of shape and values |

```python
a = np.array([1, 2])
b = np.array([3, 4])
joined = np.concatenate([a, b])            # [1, 2, 3, 4]
rows = np.stack([a, b], axis=0)            # [[1, 2], [3, 4]]
padded = np.pad(a, (2, 1), constant_values=0)   # [0, 0, 1, 2, 0]

dx = np.diff(joined)                       # [1, 1, 1]
total = np.sum(joined)
average = np.mean(joined)
largest = np.max(joined)
largest_index = np.argmax(joined)
close = np.allclose(joined, [1, 2, 3, 4])
```

`np.repeat` versus `np.tile` — `repeat` is exactly a zero-order hold on the sample index, `tile` is periodic extension:

```python
np.repeat([1, 2], 3)      # [1, 1, 1, 2, 2, 2]   each element held
np.tile([1, 2], 3)        # [1, 2, 1, 2, 1, 2]   the block repeated
```

Sampling uses:

```python
from sampling_skeleton import zero_pad

x = np.array([1.0, -1.0, 0.5])
x_padded = zero_pad(x, left=2, right=3)

t = np.arange(6) / 10.0
T_estimated = np.mean(np.diff(t))
uniform = np.allclose(np.diff(t), T_estimated)
peak_index = np.argmax(np.abs(x))
```

Zero padding adds stored zeros; it does not create new information and it does not increase the actual sampling rate. In a spectrum it only interpolates the frequency grid more finely, it does not improve resolution.

Never compare floats with `==`. `np.allclose(a, b, rtol=1e-5, atol=1e-8)` returns one bool for the whole array; `np.isclose` returns an elementwise mask.

### 4.7 Sorting, searching, rounding, and clipping

| Call | Meaning |
|---|---|
| `np.sort(a)` | sorted **copy** (`a.sort()` sorts in place) |
| `np.argsort(a)` | indices that would sort `a` |
| `np.argmax(a)` / `np.argmin(a)` | index of the largest / smallest value |
| `np.searchsorted(a, v, side="left")` | insertion index into a **sorted** `a` |
| `np.unique(a)` | sorted unique values; `return_counts=True` adds counts |
| `np.clip(a, lo, hi)` | limit values to a range |
| `np.round(a, decimals=0)` | round (banker's rounding at exact halves) |
| `np.floor`, `np.ceil`, `np.trunc` | round down / up / toward zero |
| `np.sign(a)` | -1, 0 or +1 |
| `np.maximum(a, b)` / `np.minimum(a, b)` | elementwise pairwise max / min |

```python
amplitude = np.array([0.2, 1.0, 0.4])
frequencies = np.array([5.0, 12.0, 30.0])

order = np.argsort(amplitude)[::-1]           # strongest first
ranked = list(zip(frequencies[order], amplitude[order]))
peak_frequency = frequencies[np.argmax(amplitude)]    # 12.0
```

`np.argsort` is the array answer to Python's `sorted(..., key=...)`: sort one array and reorder the others by the same index array.

`np.searchsorted` is what makes the zero-order hold fast — it finds, for every output time, which sample interval it falls into:

```python
indices = np.searchsorted(t_samples, t_output, side="right") - 1
```

`side="right"` means "a tie goes after the existing entry", which gives the half-open convention `[nT, (n+1)T)`. `side="left"` would give `(nT, (n+1)T]`. The array being searched must be sorted, which is exactly why the skeleton rejects non-increasing sample times.

```python
np.clip([-2.0, 0.5, 3.0], 0.0, 1.0)     # [0.0, 0.5, 1.0]
np.round([0.5, 1.5, 2.5])               # [0., 2., 2.]  -- halves go to even
np.unique([5.0, 12.0, 5.0], return_counts=True)   # (array([5., 12.]), array([2, 1]))
```

### 4.8 Mathematical functions

All of these are elementwise and return a new array.

| Group | Functions |
|---|---|
| trigonometric | `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`, `np.arctan`, `np.arctan2(y, x)` |
| exponential / log | `np.exp`, `np.log`, `np.log10`, `np.log2`, `np.expm1`, `np.log1p` |
| powers | `np.sqrt`, `np.square`, `np.power(a, b)`, `np.cbrt` |
| magnitude | `np.abs`, `np.absolute`, `np.fabs` |
| special | `np.sinc`, `np.sign`, `np.heaviside(x, half)` |
| constants | `np.pi`, `np.e`, `np.inf`, `np.nan` |
| integration | `np.trapezoid(y, x)` (`np.trapz` in NumPy 1.x) |
| phase | `np.angle`, `np.unwrap` |

```python
t = np.linspace(0.0, 1.0, 1001)
sine = np.sin(2 * np.pi * 3 * t)
cosine = np.cos(2 * np.pi * 3 * t)
decay = np.exp(-2 * t)
magnitude = np.abs(decay * np.exp(1j * 2 * np.pi * 3 * t))
sinc_values = np.sinc(t)
```

`np.sinc` is the **normalized** sinc:

```text
np.sinc(u) = sin(pi*u)/(pi*u),  np.sinc(0) = 1
```

Its zeros are at every nonzero integer `u`, which is exactly what makes `np.sinc((t - n*T)/T)` equal 1 at `t = nT` and 0 at every other sample instant. If you write `np.sin(x)/x` yourself you get the unnormalized sinc **and** a divide-by-zero warning at the origin; use `np.sinc`.

`np.arctan2(y, x)` gets the quadrant right where `np.arctan(y/x)` cannot; it is what `np.angle` uses internally.

### 4.9 Complex numbers

The DFT returns complex values, so these matter.

```python
z = np.exp(1j * 2 * np.pi * 5 * t)       # 1j is Python's imaginary unit

np.abs(z)          # magnitude
np.angle(z)        # phase in radians, in (-pi, pi]
np.angle(z, deg=True)
np.unwrap(np.angle(z))    # removes the 2*pi jumps, for a readable phase plot
np.real(z), np.imag(z)    # or z.real, z.imag
np.conj(z)                # complex conjugate
z.astype(complex)
np.iscomplexobj(z)        # True
```

Two things to watch:

- Plotting a complex array raises `ComplexWarning` and drops the imaginary part. Always plot `np.abs(...)` or `np.angle(...)` explicitly.
- Assigning a complex value into a float array silently discards the imaginary part. Create the array with `dtype=complex` if it must hold complex data.

For a real signal the spectrum is conjugate-symmetric: `X[-k] = conj(X[k])`, which is why a one-sided spectrum loses nothing and why amplitudes are doubled when folding to one side (section 4.12).

### 4.10 Interpolation and polynomials

| Call | Meaning |
|---|---|
| `np.interp(x_new, x, y)` | **linear** interpolation on sorted `x` |
| `np.polyfit(x, y, deg)` | least-squares polynomial coefficients, highest power first |
| `np.polyval(coeffs, x)` | evaluate a polynomial |
| `np.poly1d(coeffs)` | callable polynomial object |

```python
f_new = np.linspace(0.0, 10.0, 501)
mag_new = np.interp(f_new, f_coarse, mag_coarse, left=0.0, right=0.0)
```

`np.interp` requires `x` to be increasing, and `left`/`right` set the value outside the given range (the endpoint value by default). The skeleton uses it in `spectrum_replicas` to place each shifted copy of a baseband spectrum onto one common frequency grid.

`np.interp` is *linear* interpolation, which is the first-order hold. That is outside this syllabus as a reconstruction method — use it as a plotting tool, and use `sinc_reconstruct` or `zero_order_hold` when a question asks for reconstruction.

### 4.11 Convolution: `np.convolve`

```python
np.convolve(a, v, mode="full")
```

Computes the discrete convolution `(a * v)[n] = sum_k a[k] v[n-k]`. It is commutative, it accepts only one-dimensional arrays, and the `mode` sets which part of the result is kept:

| `mode` | Output length (with `len(a)=N`, `len(v)=M`) | Meaning |
|---|---|---|
| `"full"` | `N + M - 1` | every overlap, including the partial ones at both ends |
| `"same"` | `max(N, M)` | the centre of `full`, so the output is as long as the longer input |
| `"valid"` | `max(N, M) - min(N, M) + 1` | only the positions with complete overlap, no edge effects |

```python
x = np.array([1.0, 2.0, 3.0])
h = np.array([1.0, 1.0])

np.convolve(x, h, mode="full")     # [1., 3., 5., 3.]   length 3+2-1 = 4
np.convolve(x, h, mode="same")     # [1., 3., 5.]       length 3
np.convolve(x, h, mode="valid")    # [3., 5.]           length 3-2+1 = 2
```

Why it belongs in a sampling test: reconstruction *is* a convolution. The lecture writes

```text
xr(t) = sum_n x[n] h(t - nT)
```

which on a grid `factor` times finer than `T` is exactly "zero-stuff the samples, then convolve with the kernel `h`":

```python
from sampling_skeleton import (
    interpolate_by_convolution, sinc_kernel, upsample, zoh_kernel,
)

factor = 20
stuffed = upsample(x_samples, factor)              # impulse-train model
x_zoh = np.convolve(stuffed, zoh_kernel(factor))[: stuffed.size]
```

`upsample` inserts `factor - 1` zeros after every sample; the resulting array is the array model of `xp(t) = sum_n x(nT) delta(t - nT)`. Convolving it with a rect of length `factor` holds each sample for one period — a zero-order hold. Convolving it with a sampled sinc performs ideal interpolation.

`interpolate_by_convolution(values, factor, kernel, delay)` does the zero-stuffing, the convolution and the trimming in one call. `delay` is the kernel's centre index, which is what re-aligns the output with the input:

```python
x_zoh = interpolate_by_convolution(x_samples, factor, zoh_kernel(factor))
# ZOH kernel is causal, so delay = 0 (the default)

kernel = sinc_kernel(factor, half_width=12)
x_sinc = interpolate_by_convolution(x_samples, factor, kernel, delay=12 * factor)
# sinc kernel is symmetric with 12*factor taps on each side of its centre

assert np.allclose(x_sinc[::factor], x_samples, atol=1e-9)
```

The dense time axis that matches either output is

```python
t_dense = t_samples[0] + np.arange(x_zoh.size) / (fs * factor)
```

Two reminders. A truncated sinc kernel is only an approximation of the infinite ideal sum, so error is largest at the ends of the record. And convolution in time is multiplication in frequency — the same duality the lecture uses when it multiplies by an impulse train in time and gets shifted copies in frequency.

`np.correlate(a, v, mode=...)` has the same three modes but does not flip the second argument; it is cross-correlation, not convolution. Do not reach for it by accident.

### 4.12 The FFT family: `np.fft`

The DFT is a numerical tool here. The lecture's spectra are continuous-time Fourier transforms of infinite signals; an FFT of a finite record is an approximation of them, useful for *showing* a peak or an alias, not for proving the theory.

| Call | Returns |
|---|---|
| `np.fft.fft(x, n=None)` | full complex DFT, length `n` (default `len(x)`) |
| `np.fft.ifft(X)` | inverse of `fft`, complex |
| `np.fft.rfft(x, n=None)` | DFT of a **real** input, only the `n//2 + 1` nonnegative bins |
| `np.fft.irfft(X, n=len(x))` | inverse of `rfft`; **pass `n`** or an odd-length signal comes back one sample short |
| `np.fft.fftfreq(n, d)` | frequency of each `fft` bin, in Hz, with `d = 1/fs` |
| `np.fft.rfftfreq(n, d)` | frequency of each `rfft` bin |
| `np.fft.fftshift(a)` | reorder so the axis runs `-fs/2 ... +fs/2` |
| `np.fft.ifftshift(a)` | undo `fftshift` |
| `np.fft.fft2`, `np.fft.ifft2` | two-dimensional versions (not needed here) |

Bin layout. `np.fft.fft` returns bin `k` for `k = 0 ... N-1`, where bin `k` holds frequency `k*fs/N` for `k < N/2` and the **negative** frequency `(k - N)*fs/N` above that. `fftfreq` returns exactly those numbers, and `fftshift` puts them in ascending order for plotting:

```python
np.fft.fftfreq(8, d=1/8)     # [ 0.  1.  2.  3. -4. -3. -2. -1.]
np.fft.fftshift(np.fft.fftfreq(8, d=1/8))
                             # [-4. -3. -2. -1.  0.  1.  2.  3.]
```

Bin spacing is `fs/N = 1/(N*T) = 1/record_duration`. To resolve two tones `df` apart you need a record at least `1/df` seconds long; zero padding makes the grid finer but does not add resolution.

Amplitude scaling. `rfft` returns unnormalized sums, so divide by `N` and double the bins that stand for a positive/negative pair. DC (bin 0) and, for even `N`, the Nyquist bin have no partner and must not be doubled:

```python
X = np.fft.rfft(x)
f = np.fft.rfftfreq(x.size, d=1 / fs)
amplitude = 2 * np.abs(X) / x.size
amplitude[0] /= 2
if x.size % 2 == 0:
    amplitude[-1] /= 2
```

That is exactly what the skeleton's `magnitude_spectrum` does:

```python
from sampling_skeleton import magnitude_spectrum

fs = 100.0
t = np.arange(int(fs)) / fs
x = np.sin(2 * np.pi * 12 * t)
f, amplitude = magnitude_spectrum(x, fs)
print(f[np.argmax(amplitude)])        # 12.0 Hz
```

Use `remove_mean=True` when a large DC offset would dwarf the tones you want to see.

A two-sided picture, which is the one that matches the lecture's drawings, needs `fft` plus `fftshift`:

```python
from sampling_skeleton import two_sided_spectrum

f, X = two_sided_spectrum(x, fs)      # f ascends from -fs/2
plt.plot(f, np.abs(X))
```

A real signal gives a spectrum that is symmetric about 0, so the 12 Hz tone shows up at both -12 and +12 Hz.

Spectral leakage. A tone whose frequency is not an exact multiple of `fs/N` does not land on a bin centre, and its energy smears across neighbours. Choose a record holding a whole number of cycles (`N = fs/f` an integer) when you want a clean single spike, and say "leakage" rather than "aliasing" when you see skirts around a peak.

Seeing an alias:

```python
fs = 10.0
t = np.arange(50) / fs
x = np.sin(2 * np.pi * 9.0 * t)       # 9 Hz sampled at 10 Hz
f, amplitude = magnitude_spectrum(x, fs)
print(f[np.argmax(amplitude)])        # 1.0 -- the folded frequency, not 9.0
```

The FFT cannot show you the original 9 Hz, because after sampling it genuinely is not there any more. That is the point of the theorem.

Drawing the replication picture directly — the lecture's `Xp = (1/T) sum_k X(f - k*fs)`:

```python
from sampling_skeleton import spectrum_replicas

f_base = np.linspace(-4.0, 4.0, 401)
mag_base = np.clip(1.0 - np.abs(f_base) / 4.0, 0.0, None)   # triangular X(f)

grid, stack, total = spectrum_replicas(f_base, mag_base, fs=20.0, copies=2)
for row in stack:
    plt.plot(grid, row, color="C0", alpha=0.4)
plt.plot(grid, total, color="C3", label="sum of copies")
```

With `fs = 20` and `f_max = 4` the copies sit clear of one another and a low-pass filter can pick out the centre one. Rerun with `fs = 6` and the copies overlap; `total` is then larger than any single copy in the overlap region, and no filter can undo that. This is aliasing, drawn.

`np.fft.fft` is fastest when `N` is a power of two or a product of small primes, which matters only for very long records.

### 4.13 Random numbers, saving, and printing

```python
rng = np.random.default_rng(seed=0)         # the modern generator
noise = rng.normal(0.0, 0.1, size=t.size)   # Gaussian noise
uniform = rng.uniform(-1.0, 1.0, size=10)
picks = rng.integers(0, 10, size=5)
rng.shuffle(values)                         # in place
```

Always seed the generator when a grader needs to reproduce your figure. The old `np.random.seed` / `np.random.rand` interface still works but `default_rng` is preferred.

```python
np.savetxt("samples.csv", np.column_stack([t, x]), delimiter=",", header="t,x")
data = np.loadtxt("samples.csv", delimiter=",", skiprows=1)
t, x = data[:, 0], data[:, 1]

np.save("x.npy", x)          # binary, exact
x = np.load("x.npy")
```

```python
np.set_printoptions(precision=3, suppress=True)   # readable console output
print(np.array2string(x[:5], precision=2))
```

`suppress=True` stops values like `1.2e-17` being printed in scientific notation, which makes "is this zero?" much easier to judge.

### 4.14 Errors you will actually see

| Message | Cause | Fix |
|---|---|---|
| `ValueError: operands could not be broadcast together with shapes (100,) (99,)` | time and signal arrays have different lengths | check `t.shape == x.shape` |
| `ValueError: truth value of an array ... is ambiguous` | `and` / `or` / `if` on an array | use `&`, `|`, `~`, or `np.any` / `np.all` |
| `ValueError: cannot reshape array of size 6 into shape (4,)` | the new shape does not match `a.size` | use `-1` for the inferred axis |
| `IndexError: index 100 is out of bounds for axis 0 with size 100` | off-by-one | the last index is `n-1`, or use `x[-1]` |
| `TypeError: only length-1 arrays can be converted to Python scalars` | `math.sin` on an array | use `np.sin` |
| `RuntimeWarning: invalid value encountered in divide` | `0/0` somewhere; produces `nan` | guard the denominator, or use `np.sinc` |
| `ComplexWarning: Casting complex values ... discarding the imaginary part` | plotting or storing a complex array as float | plot `np.abs(...)` or `np.angle(...)` |

### 4.15 Quick function index

**Creation** `array` `asarray` `arange` `linspace` `zeros` `ones` `full` `empty` `zeros_like` `ones_like` `full_like` `eye`

**Shape** `shape` `ndim` `size` `dtype` `astype` `copy` `reshape` `ravel` `flatten` `transpose` `newaxis` `expand_dims` `squeeze` `tolist`

**Combine** `concatenate` `stack` `vstack` `hstack` `column_stack` `pad` `tile` `repeat` `roll` `flip` `append` `insert` `delete`

**Select** `where` `select` `piecewise` `nonzero` `flatnonzero` `take` `compress` `clip`

**Reduce** `sum` `mean` `std` `var` `min` `max` `ptp` `argmin` `argmax` `cumsum` `cumprod` `any` `all` `count_nonzero` `trapezoid`

**Sort / search** `sort` `argsort` `searchsorted` `unique`

**Math** `sin` `cos` `tan` `arctan2` `exp` `log` `log10` `sqrt` `square` `power` `abs` `sign` `floor` `ceil` `round` `trunc` `sinc` `heaviside` `maximum` `minimum` `pi` `e` `inf` `nan`

**Complex** `real` `imag` `abs` `angle` `unwrap` `conj` `iscomplexobj`

**Compare** `isclose` `allclose` `array_equal` `isnan` `isinf` `isfinite`

**Signals** `convolve` `correlate` `interp` `polyfit` `polyval`

**Fourier** `fft.fft` `fft.ifft` `fft.rfft` `fft.irfft` `fft.fftfreq` `fft.rfftfreq` `fft.fftshift` `fft.ifftshift`

**Random / IO** `random.default_rng` `savetxt` `loadtxt` `save` `load` `set_printoptions`

## 5. Matplotlib Essentials

Marks in a plotting question come from the right *kind* of plot (continuous curve versus discrete stems versus a held staircase), plus axis labels, a title and a legend. Section 5.9 is the index.

### 5.1 Figure and axes anatomy

There are two styles. The object-oriented one is used throughout this guide because it survives subplots and returns objects you can keep customizing.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))   # OO style: one Figure, one Axes
ax.plot(t, x)
ax.set_xlabel("Time (s)")
plt.show()
```

```python
plt.plot(t, x)              # pyplot style: acts on the "current" axes
plt.xlabel("Time (s)")
plt.show()
```

`plt.subplots(nrows=1, ncols=1, figsize=(w, h), sharex=False, sharey=False)` returns `(fig, ax)` for a single panel and `(fig, axes_array)` for several. The `Figure` is the whole canvas; each `Axes` is one plot box with its own data, labels and legend. `ax.figure` gets back to the figure — which is how the skeleton's `plot_signal` accepts an existing `ax`.

If you run a script from a terminal, `plt.show()` blocks until the window closes. Under a non-interactive backend nothing appears at all — save instead:

```python
import matplotlib
matplotlib.use("Agg")       # must come BEFORE importing pyplot
```

### 5.2 Line plots: `ax.plot`

```python
ax.plot(x, y, fmt, **kwargs)
```

Draws points joined by straight lines. Use it for a dense reference curve that *looks* continuous.

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, x, color="C0", linewidth=2, label="x(t)")
ax.set(title="Signal", xlabel="Time (s)", ylabel="Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
```

Common keywords:

| Keyword | Values |
|---|---|
| `color` | `"C0"`…`"C9"` (the cycle), `"red"`, `"#1f77b4"` |
| `linestyle` / `ls` | `"-"` solid, `"--"` dashed, `":"` dotted, `"-."` dash-dot, `"none"` |
| `linewidth` / `lw` | number of points |
| `marker` | `"o"`, `"."`, `"x"`, `"s"`, `"^"`, `"none"` |
| `markersize` / `ms` | number of points |
| `alpha` | 0 transparent to 1 opaque |
| `label` | legend text |
| `zorder` | draw order, higher is on top |

The `fmt` shorthand packs colour, marker and linestyle into one string: `"r--"` is a red dashed line, `"C1o"` is orange circles with no line, `"k-"` is a black solid line.

```python
ax.plot(t_output, x_reconstructed, "--", label="reconstruction")
```

One `plot` call can draw several lines (`ax.plot(t, x1, t, x2)`), and it returns the list of `Line2D` objects it created.

### 5.3 Discrete samples: `ax.stem`

```python
ax.stem(x, y, linefmt="C1-", markerfmt="C1o", basefmt="k-", label="samples")
```

Draws a vertical line and a marker per sample. This is the correct picture for a discrete-time sequence — a sampled signal plotted with `plot` wrongly suggests values exist between the samples.

```python
fig, ax = plt.subplots()
ax.stem(t_samples, x_samples, linefmt="C1-", markerfmt="C1o", basefmt="k-")
ax.set(xlabel="Time (s)", ylabel="x[n]", title="Samples")
ax.grid(True, alpha=0.3)
plt.show()
```

`linefmt`, `markerfmt` and `basefmt` are `fmt` strings for the stems, the tips and the baseline. `basefmt=" "` hides the baseline. `stem` gets slow and unreadable past a few hundred points — stem the samples, plot the dense curve.

### 5.4 Held and filled shapes: `step`, `fill_between`, `scatter`, `bar`

```python
ax.step(t_output, x_zoh, where="post", label="ZOH")
```

`ax.step(x, y, where=...)` draws a staircase, which is exactly a zero-order hold. **`where="post"`** holds each value until the *next* x — the `[nT, (n+1)T)` convention used by the skeleton. `where="pre"` holds from the previous x, and `where="mid"` puts the jump halfway. Using the wrong one shifts the whole staircase by a sample period and is an easy mark to lose.

```python
ax.fill_between(f, 0, magnitude, alpha=0.3)          # shade under a curve
ax.fill_between(t, -1, 1, where=(t > 0.5), alpha=0.2, color="C2")   # a region
ax.scatter(t_samples, x_samples, s=30, zorder=3)     # markers only, no stems
ax.bar(frequencies, amplitudes, width=0.4)           # bar chart of peaks
ax.axhline(0.0, color="k", linewidth=0.8)            # baseline
```

`fill_between` is the readable way to shade the guard band between spectral copies, or the overlap region that causes aliasing.

### 5.5 Reference lines, text, and annotations

```python
ax.axvline(fs / 2, color="C3", linestyle="--", label="fs/2")
ax.axhline(1.0, color="gray", linewidth=0.8)
ax.axvspan(-fs / 2, fs / 2, alpha=0.15, color="C2", label="passband")
ax.axhspan(-1, 1, alpha=0.1)

ax.text(0.5, 1.05, "f_max = 12 Hz", transform=ax.transAxes)   # axes coordinates
ax.annotate(
    "alias at 1 Hz",
    xy=(1.0, 0.9),                     # the point being labelled (data coords)
    xytext=(3.0, 1.1),                 # where the text sits
    arrowprops={"arrowstyle": "->"},
)
```

`axvline` / `axhline` span the whole axes regardless of the data limits — ideal for marking `fs/2`, `fs`, the first ZOH null, or a Nyquist boundary. `transform=ax.transAxes` makes `(0, 0)` the bottom-left and `(1, 1)` the top-right of the box, which is how to place text that should not move when the data changes.

### 5.6 Axis configuration

```python
ax.set(title="...", xlabel="Time (s)", ylabel="Amplitude")   # several at once
ax.set_title("...")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|X(f)|")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(left=0)                 # one side only

ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks(t_samples, minor=True)
ax.set_xticklabels(["0", "T/4", "T/2", "3T/4", "T"])
ax.tick_params(axis="both", labelsize=9)

ax.grid(True, alpha=0.3)
ax.grid(True, which="both", axis="x")

ax.legend()                                   # uses each artist's label=
ax.legend(["reference", "samples"])           # or explicit labels, in draw order
ax.legend(loc="upper right", ncol=2, framealpha=0.9)

ax.set_yscale("log")                          # or ax.semilogy(f, magnitude)
ax.set_aspect("equal")
ax.invert_yaxis()
```

Two legend habits worth keeping: pass `label=` to every artist and call `ax.legend()` with no arguments, or pass the list — do not mix the two, because a list overrides the labels positionally and mislabels anything you forgot to count. Note that `ax.stem` needs its `label=` too.

Setting `ax.set_xlim(0, fs / 2)` on a one-sided spectrum is what makes the peak readable; without it the plot is mostly empty high-frequency space.

### 5.7 Overlaying a reference and its samples

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_dense, x_dense, label="dense reference")
ax.stem(t_samples, x_samples, linefmt="C1-", markerfmt="C1o", basefmt="k-")
ax.set_xlim(t_dense[0], t_dense[-1])
ax.set_ylim(-1.2, 1.2)
ax.set(xlabel="Time (s)", ylabel="Amplitude")
ax.legend(["dense reference", "samples"])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Or use the skeleton:

```python
fig, ax = plot_sampling(t_dense, x_dense, t_samples, x_samples)
plt.show()
```

A dense plotted curve is still a finite array. It only looks continuous on the screen — say so if the question asks.

A spectrum with its Nyquist markers:

```python
from sampling_skeleton import magnitude_spectrum, plot_spectrum

f, amplitude = magnitude_spectrum(x_samples, fs)
fig, ax = plot_spectrum(f, amplitude, discrete=True, fs=fs, title="Spectrum")
ax.set_xlim(0, fs / 2)
plt.show()
```

`plot_spectrum(..., fs=fs)` adds the dashed `fs/2` line and the dotted `fs` line for you; leave `fs` out and it plots the axes alone.

### 5.8 Subplots, layout, and saving

```python
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(t_dense, x_dense)
axes[0].set_title("Reference")
axes[1].stem(t_samples, x_samples)
axes[1].set_title("Samples")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Amplitude")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Sampling at 25 Hz")
fig.tight_layout()
fig.savefig("sampling_result.png", dpi=200, bbox_inches="tight")
plt.show()
```

With `nrows > 1` and `ncols > 1`, `axes` is a 2-D array indexed `axes[row, col]`; `axes.flat` iterates over all of them. `sharex=True` links the x-axes so zooming one zooms all, and hides the duplicate tick labels.

| Call | Purpose |
|---|---|
| `fig.tight_layout()` | stop labels overlapping |
| `fig.suptitle(...)` | one title over all panels |
| `fig.savefig(path, dpi=200, bbox_inches="tight")` | write PNG/PDF/SVG by extension |
| `plt.show()` | display; call it **after** every plotting call |
| `plt.close(fig)` / `plt.close("all")` | free memory in a loop |
| `fig.add_subplot(2, 1, 1)` | add panels one at a time |
| `plt.figure(figsize=...)` | new figure in pyplot style |

Save before `show()` in a script: some backends clear the figure once the window closes, producing a blank file.

The helper `plot_signal(t, x, discrete=False)` returns `(fig, ax)` without calling `plt.show()`, so it stays easy to customize:

```python
fig, ax = plot_signal(t, x, label="x(t)", title="Signal")
ax.axvline(0.5, color="C3", linestyle="--")
ax.set_ylim(-2, 2)
plt.show()
```

### 5.9 Quick function index

**Figure / axes** `plt.subplots` `plt.figure` `fig.add_subplot` `fig.suptitle` `fig.tight_layout` `fig.savefig` `plt.show` `plt.close` `matplotlib.use`

**Drawing** `ax.plot` `ax.stem` `ax.step` `ax.scatter` `ax.bar` `ax.fill_between` `ax.hist` `ax.imshow`

**Reference marks** `ax.axvline` `ax.axhline` `ax.axvspan` `ax.axhspan` `ax.text` `ax.annotate`

**Axes setup** `ax.set` `ax.set_title` `ax.set_xlabel` `ax.set_ylabel` `ax.set_xlim` `ax.set_ylim` `ax.set_xticks` `ax.set_xticklabels` `ax.tick_params` `ax.grid` `ax.legend` `ax.set_yscale` `ax.semilogy` `ax.invert_yaxis`

**Plot kinds for this test** dense curve → `plot`; discrete samples → `stem`; zero-order hold → `step(where="post")`; spectrum peaks → `stem` or `plot`; shaded band → `fill_between` or `axvspan`.

## 6. Building Signals with Arrays and Masks

### 6.1 Sinusoids and sums of tones

```python
def signal(t):
    return 2 * np.cos(2 * np.pi * 5 * t + np.pi / 4) + np.sin(2 * np.pi * 12 * t)
```

Read frequencies only from the arguments multiplying `t` after converting angular frequency if necessary:

```text
cos(20*pi*t) -> 2*pi*f = 20*pi -> f = 10 Hz
sin(14*t)    -> omega = 14 rad/s -> f = 14/(2*pi) Hz
```

For the function above, `f_max = 12 Hz`, so the strict condition is `fs > 24 Hz`.

### 6.2 Unit steps and pulses

```python
t = np.linspace(-2, 2, 1001)
u = unit_step(t)
pulse = rectangular_pulse(t, start=-0.5, stop=0.5)

# A pulse made from two steps:
pulse2 = unit_step(t, at=-0.5) - unit_step(t, at=0.5)
assert np.allclose(pulse, pulse2)
```

The half-open convention `[start, stop)` prevents overlapping boundaries when adjacent piecewise intervals are combined.

### 6.3 Piecewise signals with masks

For

```text
x(t) = 0          t < -1
       t + 1      -1 <= t < 0
       1          0 <= t < 1
       exp(-(t-1)) t >= 1
```

use:

```python
t = np.linspace(-2, 3, 1001)
x = np.zeros_like(t)

m1 = (t >= -1) & (t < 0)
m2 = (t >= 0) & (t < 1)
m3 = t >= 1

x[m1] = t[m1] + 1
x[m2] = 1
x[m3] = np.exp(-(t[m3] - 1))
```

Alternatively:

```python
x = np.piecewise(
    t,
    [t < -1, m1, m2, m3],
    [0, lambda z: z + 1, 1, lambda z: np.exp(-(z - 1))],
)
```

Masks are often easier to debug in a timed test.

## 7. Uniform Sampling Workflow

### 7.1 Standard template

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import make_time_axis, plot_sampling, sample_signal


def x_of_t(t):
    return np.cos(2 * np.pi * 4 * t) + 0.5 * np.sin(2 * np.pi * 9 * t)


fs = 25.0
start, stop = 0.0, 1.0

t_dense = np.linspace(start, stop, 2001)
x_dense = x_of_t(t_dense)
t_samples, x_samples = sample_signal(x_of_t, fs, start, stop)

print("number of samples:", len(t_samples))
print("T:", 1 / fs)
fig, ax = plot_sampling(t_dense, x_dense, t_samples, x_samples)
plt.show()
```

Here `f_max = 9 Hz`, and `25 > 18`, so the theorem permits ideal recovery.

### 7.2 Sample count and endpoints

For a half-open interval of duration `D` with integer `D*fs`:

```text
N = D*fs
t[n] = start + n/fs, n = 0, 1, ..., N-1
```

Example: one second at `fs=10 Hz` gives ten samples at `0.0, 0.1, ..., 0.9`. Including `t=1.0` gives eleven displayed points, but there are still only ten sampling intervals in that one-second duration.

Use `make_time_axis(start, stop, fs, endpoint=True)` only when the endpoint is explicitly required and lies on the grid.

## 8. Sampling Theorem and Aliasing

### 8.1 Choosing a rate

```python
frequencies_hz = np.array([5.0, 12.0, 30.0])
f_max = frequencies_hz.max()
minimum_rate = nyquist_rate(f_max)

print(minimum_rate)                    # 60.0
print(meets_nyquist(60.0, f_max))      # False: strict lecture condition
print(meets_nyquist(61.0, f_max))      # True
```

If a setter asks for the Nyquist rate, answer `60 Hz`. If asked for a sampling rate satisfying the lecture's condition, answer any `fs > 60 Hz`, such as `61 Hz` or a convenient higher rate.

### 8.2 Predicting aliases

For a real sinusoid, fold its frequency into `[0, fs/2]`:

```python
print(alias_frequency(9.0, fs=10.0))   # 1.0
print(alias_frequency(14.0, fs=10.0))  # 4.0
```

Manual method: choose an integer `k` so that

```text
f_alias = |f - k*fs|
```

falls in `[0, fs/2]`.

### 8.3 Same samples, different continuous signals

If two complex-tone frequencies differ by an integer multiple of `fs`, their sample phases differ by an integer multiple of `2*pi*n`:

```python
fs = 10.0
t_samples = np.arange(10) / fs
x1 = np.sin(2 * np.pi * 3 * t_samples)
x2 = np.sin(2 * np.pi * 13 * t_samples)
print(np.allclose(x1, x2))  # True
```

The continuous curves differ between sample locations, showing why samples alone are ambiguous without a band-limit assumption.

### 8.4 Spectrum-copy picture

Sampling produces shifted copies at `k*fs` in hertz, or `k*omega_s` in rad/s. A finite NumPy array cannot store ideal Dirac impulses. For coding questions, plot conceptual copy boundaries or plot a finite-record FFT and explicitly call it an approximation.

Oversampling leaves a guard band between copies. Undersampling overlaps copies and causes aliasing.

To draw the picture rather than describe it, give `spectrum_replicas` any baseband magnitude shape and let it place, scale and sum the copies:

```python
from sampling_skeleton import spectrum_replicas

f_base = np.linspace(-4.0, 4.0, 401)
mag_base = np.clip(1.0 - np.abs(f_base) / 4.0, 0.0, None)   # f_max = 4 Hz

for fs, ax in zip([20.0, 6.0], plt.subplots(2, 1, figsize=(9, 6))[1]):
    grid, stack, total = spectrum_replicas(f_base, mag_base, fs, copies=2)
    for row in stack:
        ax.plot(grid, row, color="C0", alpha=0.4)
    ax.plot(grid, total, color="C3", label="sum of copies")
    ax.axvline(fs / 2, color="k", linestyle="--", label="fs/2")
    ax.set_title(f"fs = {fs} Hz, f_max = 4 Hz")
    ax.legend()
plt.tight_layout()
plt.show()
```

At `fs = 20` the copies are separated by a guard band and the centre one can be filtered out intact. At `fs = 6` neighbouring copies overlap, `total` exceeds every individual copy inside the overlap, and no filter can separate them again. Each copy is scaled by `1/T = fs`, matching `Xp = (1/T) sum_k X(f - k*fs)`.

This is still a drawing built from arrays, not a transform of Dirac impulses. Say so if the question asks what is on the axes.

## 9. Ideal Sinc Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, sinc_reconstruct


def signal(t):
    return np.sin(2 * np.pi * 3 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, -1.0, 1.1)
t_output = np.linspace(-1.0, 1.0, 2001)
x_reconstructed = sinc_reconstruct(t_samples, x_samples, t_output)

plt.plot(t_output, signal(t_output), label="original")
plt.plot(t_output, x_reconstructed, "--", label="finite sinc reconstruction")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Vectorized core:

```python
T = 1 / fs
kernel = np.sinc((t_output[:, None] - t_samples[None, :]) / T)
x_reconstructed = kernel @ x_samples
```

Each column is one shifted sinc basis function. The theoretical sum is infinite. A finite record truncates it, so error is usually largest near the record edges. At the supplied sample locations, normalized sinc interpolation returns the sample values exactly (apart from roundoff).

The same reconstruction as a convolution. `xr(t) = sum_n x[n] h(t - nT)` is a filtering operation, so zero-stuffing the samples onto a finer grid and convolving with a sampled sinc gives the identical result (section 4.11):

```python
from sampling_skeleton import interpolate_by_convolution, sinc_kernel

factor = 20                      # output grid is 20x finer than T
half_width = 12                  # keep 12 sample periods of the sinc each side
kernel = sinc_kernel(factor, half_width)
x_sinc = interpolate_by_convolution(x_samples, factor, kernel,
                                    delay=half_width * factor)
t_sinc = t_samples[0] + np.arange(x_sinc.size) / (fs * factor)

assert np.allclose(x_sinc[::factor], x_samples, atol=1e-9)
```

Use the matrix form when you want reconstruction at an arbitrary set of output times, and the convolution form when the output grid is a uniform refinement of the sample grid — it is the version that shows reconstruction as low-pass filtering.

## 10. Zero-Order Hold

### 10.1 Time-domain reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, zero_order_hold


def signal(t):
    return np.sin(2 * np.pi * 2 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.1)
t_output = np.linspace(0.0, 1.0, 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_output)

plt.plot(t_output, signal(t_output), label="original")
plt.step(t_output, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

The helper holds `x[n]` over `[nT, (n+1)T)`. It uses a fill value outside the recorded sample range instead of silently extrapolating.

Without the helper:

```python
indices = np.searchsorted(t_samples, t_output, side="right") - 1
valid = (t_output >= t_samples[0]) & (t_output <= t_samples[-1])
x_zoh = np.full(t_output.shape, np.nan)
x_zoh[valid] = x_samples[indices[valid]]
```

As a convolution, which is the form that matches `h0(t) = 1 on [0, T)`:

```python
from sampling_skeleton import interpolate_by_convolution, zoh_kernel

factor = 20
x_zoh = interpolate_by_convolution(x_samples, factor, zoh_kernel(factor))
t_zoh = t_samples[0] + np.arange(x_zoh.size) / (fs * factor)

assert np.allclose(x_zoh, np.repeat(x_samples, factor))
```

The kernel is `factor` ones — the rect `h0` sampled on the finer grid. Because it is causal, its delay is zero, so no re-alignment is needed. On that uniform grid the result is identical to `np.repeat`, which is the cheapest way to say "hold each sample".

### 10.2 Frequency response and droop

```python
from sampling_skeleton import zoh_frequency_response

T = 0.1
f = np.linspace(-20.0, 20.0, 4001)
H0 = zoh_frequency_response(f, T)

plt.plot(f, np.abs(H0))
plt.axvline(1 / T, color="C1", linestyle="--", label="first null: fs")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H0(f)|")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

print(abs(zoh_frequency_response([0.0], T)[0]))   # 0.1
print(abs(zoh_frequency_response([10.0], T)[0]))  # approximately 0
```

ZOH is practical but not ideal: its passband gain is not flat, and it does not completely remove shifted spectral copies.

## 11. Copy-Ready Problem Templates

### Template A: define, sample, and overlay

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import plot_sampling, sample_signal


def signal(t):
    return np.cos(2 * np.pi * F1 * t) + A2 * np.sin(2 * np.pi * F2 * t)


fs = FS
t_dense = np.linspace(START, STOP, 2001)
t_samples, x_samples = sample_signal(signal, fs, START, STOP)
fig, ax = plot_sampling(t_dense, signal(t_dense), t_samples, x_samples)
plt.show()
```

Replace the uppercase names with numbers.

### Template B: piecewise signal

```python
t = np.linspace(START, STOP, 2001)
x = np.zeros_like(t)

m1 = (t >= A) & (t < B)
m2 = (t >= B) & (t <= C)
x[m1] = FORMULA_1_USING_T[m1]
x[m2] = FORMULA_2_USING_T[m2]

plot_signal(t, x)
plt.show()
```

### Template C: spectrum visualization

```python
t_samples, x_samples = sample_signal(signal, fs, START, STOP)
f, amplitude = magnitude_spectrum(x_samples, fs, remove_mean=True)
plt.stem(f, amplitude)
plt.xlim(0, fs / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Estimated amplitude")
plt.grid(alpha=0.3)
plt.show()
```

### Template D: ideal sinc reconstruction

```python
t_output = np.linspace(t_samples[0], t_samples[-1], 2001)
x_sinc = sinc_reconstruct(t_samples, x_samples, t_output)
plt.plot(t_output, x_sinc, label="sinc reconstruction")
plt.stem(t_samples, x_samples)
plt.legend()
plt.show()
```

### Template E: ZOH reconstruction

```python
t_output = np.linspace(t_samples[0], t_samples[-1], 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_output)
plt.step(t_output, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples)
plt.legend()
plt.show()
```

### Template F: reconstruction by convolution

```python
from sampling_skeleton import (
    interpolate_by_convolution, sinc_kernel, zoh_kernel,
)

factor = 20
t_dense = t_samples[0] + np.arange(x_samples.size * factor) / (fs * factor)

x_zoh = interpolate_by_convolution(x_samples, factor, zoh_kernel(factor))

half_width = 12
x_sinc = interpolate_by_convolution(
    x_samples, factor, sinc_kernel(factor, half_width), delay=half_width * factor
)

plt.plot(t_dense, x_zoh, label="ZOH")
plt.plot(t_dense, x_sinc, "--", label="ideal sinc")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Template G: spectrum replication picture

```python
from sampling_skeleton import spectrum_replicas

f_base = np.linspace(-F_MAX, F_MAX, 401)
mag_base = np.clip(1.0 - np.abs(f_base) / F_MAX, 0.0, None)

grid, stack, total = spectrum_replicas(f_base, mag_base, FS, copies=2)
for row in stack:
    plt.plot(grid, row, color="C0", alpha=0.4)
plt.plot(grid, total, color="C3", label="sum of copies")
plt.axvline(FS / 2, color="k", linestyle="--", label="fs/2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## 12. Common Mistakes and Debugging Checklist

- **Mixing Hz and rad/s:** `cos(omega*t)` has `f = omega/(2*pi)` Hz. `cos(2*pi*f*t)` already uses hertz.
- **Using the Nyquist rate as if strict equality were safe:** the lecture condition is `fs > 2*f_max`.
- **Calling the dense plot continuous:** it is a high-resolution numerical reference.
- **Using `linspace(..., fs)` for one second without thinking:** the third argument is number of points, not samples per second.
- **Including both endpoints accidentally:** `np.linspace(0, 1, 11)` has 11 points but ten intervals.
- **Writing `and` for masks:** use `(condition1) & (condition2)`.
- **Forgetting elementwise operations:** arrays use `*`, `/`, and `**` elementwise; do not use `math.sin` on arrays.
- **Looping unnecessarily:** define signal functions with NumPy operations.
- **Treating an FFT bin as a Dirac impulse:** it is a finite-record numerical value.
- **Forgetting finite-sinc edge error:** the exact theory assumes infinitely many samples.
- **Extending ZOH outside recorded data silently:** decide and state a fill or extrapolation rule.
- **Calling zero padding oversampling:** padding changes array length, not the data-acquisition rate.
- **Using the wrong `np.convolve` mode:** `"full"` is `N+M-1` long, `"same"` is `max(N, M)`, `"valid"` is `max(N, M)-min(N, M)+1`. Trim deliberately instead of hoping `"same"` aligns.
- **Forgetting the kernel delay:** a symmetric sinc kernel shifts the output by half its length. Pass `delay=half_width*factor`; a causal ZOH rect needs `delay=0`.
- **`ax.step(where="pre")` for a zero-order hold:** the hold runs forward, so it is `where="post"`.
- **Reading `np.fft.fft` output as ascending frequency:** bins above `N/2` are negative frequencies. Use `np.fft.fftfreq` plus `np.fft.fftshift` before plotting.
- **Skipping the FFT amplitude scaling:** divide by `N` and double the paired bins, but not DC and not the Nyquist bin.
- **Blaming aliasing for leakage:** skirts around a peak mean the tone is not on a bin centre. Choose a whole number of cycles in the record.
- **`list.sort()` used as an expression:** it sorts in place and returns `None`. Use `sorted(...)` when you want a value.

Quick diagnostics:

```python
print(t.shape, x.shape)
print("estimated T:", np.mean(np.diff(t)))
print("estimated fs:", 1 / np.mean(np.diff(t)))
print("uniform:", np.allclose(np.diff(t), np.diff(t)[0]))
print("finite:", np.all(np.isfinite(x)))
```

## 13. Practice Problems

Try these before opening the solution section.

### Problem 1: Arrays, slices, and masks

Create `n = [0, 1, ..., 11]`. Form `x[n] = n^2 - 5n`. Print the even-indexed samples, replace every negative value with zero using a mask, and zero-pad the result with two zeros on each side.

### Problem 2: Piecewise signal

On `-2 <= t <= 2`, construct

```text
x(t) = 1 - |t|, |t| <= 1
       0,       otherwise
```

using masks or `np.where`. Plot it with correct labels and limits.

### Problem 3: Safe sinusoid sampling

Let `x(t) = cos(2*pi*6*t)`. Sample it over `[0, 1)` at `fs=20 Hz`. Plot the samples over a dense reference and state whether the strict Nyquist condition holds.

### Problem 4: Rate for a multitone signal

For

```text
x(t) = 2cos(10*pi*t) + sin(60*pi*t) - 0.5cos(24*pi*t),
```

list the component frequencies, find the Nyquist rate, and choose an integer sampling rate satisfying the lecture's strict condition. Generate one second of samples.

### Problem 5: Predict and display aliasing

Sample `x(t) = sin(2*pi*9*t)` at `fs=10 Hz` for one second. Predict the nonnegative alias frequency, then compare the samples with a `1 Hz` sinusoid. Account for any sign difference.

### Problem 6: Same samples from different signals

At `fs=10 Hz`, verify numerically that `sin(2*pi*3*t)` and `sin(2*pi*13*t)` have equal samples over `[0, 1)`. Plot both dense curves and the shared samples.

### Problem 7: Ideal sinc reconstruction

Sample `sin(2*pi*3*t)` at `fs=10 Hz` from `t=-1` through `t=1` and reconstruct it on a dense grid using sinc interpolation. Verify exact agreement at sample positions and comment on the error near the record edges.

### Problem 8: Zero-order hold

Sample `cos(2*pi*2*t)` at `fs=10 Hz` over `[0, 1]`. Reconstruct it on a dense grid with ZOH and plot original, samples, and held result.

### Problem 9: ZOH frequency response

For `T=0.1 s`, compute and plot `|H0(f)|` from `-20 Hz` to `20 Hz`. Report its DC gain and first positive null.

### Problem 10: Mixed sessional problem

Let

```text
x(t) = sin(2*pi*4*t) + 0.4cos(2*pi*11*t).
```

Use `fs=30 Hz` on `[0, 1)`. Confirm Nyquist, create the samples, select samples with positive amplitude using a mask, pad the sample array with four zeros on the right, compute a numerical magnitude spectrum, and compare sinc and ZOH reconstructions on one figure.

## 14. Complete Solutions

### Solution 1

```python
import numpy as np
from sampling_skeleton import zero_pad

n = np.arange(12)
x = n**2 - 5*n
print("even indices:", x[::2])

x_nonnegative = x.copy()
x_nonnegative[x_nonnegative < 0] = 0
x_padded = zero_pad(x_nonnegative, left=2, right=2)
print(x_padded)
```

The mask modifies all negative entries at once. Padding creates a longer stored sequence but does not change a sampling rate.

### Solution 2

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-2.0, 2.0, 1001)
x = np.where(np.abs(t) <= 1.0, 1.0 - np.abs(t), 0.0)

plt.plot(t, x)
plt.xlim(-2, 2)
plt.ylim(-0.1, 1.1)
plt.xlabel("Time (s)")
plt.ylabel("x(t)")
plt.title("Triangular pulse")
plt.grid(alpha=0.3)
plt.show()
```

The condition produces one vectorized Boolean mask; `np.where` chooses the inside and outside formulas elementwise.

### Solution 3

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import meets_nyquist, plot_sampling, sample_signal


def signal(t):
    return np.cos(2 * np.pi * 6 * t)


fs = 20.0
t_dense = np.linspace(0.0, 1.0, 2001)
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(meets_nyquist(fs, 6.0))  # True because 20 > 12
plot_sampling(t_dense, signal(t_dense), t_samples, x_samples)
plt.show()
```

The strict Nyquist condition holds, so ideal recovery is possible under the band-limited model.

### Solution 4

```python
import numpy as np
from sampling_skeleton import nyquist_rate, sample_signal


def signal(t):
    return 2*np.cos(10*np.pi*t) + np.sin(60*np.pi*t) - 0.5*np.cos(24*np.pi*t)


frequencies = np.array([5.0, 30.0, 12.0])
f_max = frequencies.max()
print("Nyquist rate:", nyquist_rate(f_max))  # 60 Hz

fs = 61.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(len(x_samples))  # 61
```

The maximum component is `30 Hz`, so the Nyquist rate is `60 Hz`; the strict lecture condition requires `fs > 60 Hz`.

### Solution 5

```python
import numpy as np
from sampling_skeleton import alias_frequency

fs = 10.0
t = np.arange(10) / fs
x9 = np.sin(2 * np.pi * 9 * t)
x1 = np.sin(2 * np.pi * 1 * t)

print(alias_frequency(9, fs))       # 1.0
print(np.allclose(x9, -x1))         # True
```

The nonnegative alias frequency is `1 Hz`. For these sine phases, the `9 Hz` samples match `-sin(2*pi*1*t)`; alias frequency describes the rate, while phase/sign must still be checked.

### Solution 6

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10.0
t_samples = np.arange(10) / fs
t_dense = np.linspace(0.0, 1.0, 3001)

x3_samples = np.sin(2 * np.pi * 3 * t_samples)
x13_samples = np.sin(2 * np.pi * 13 * t_samples)
print(np.allclose(x3_samples, x13_samples))  # True

plt.plot(t_dense, np.sin(2*np.pi*3*t_dense), label="3 Hz")
plt.plot(t_dense, np.sin(2*np.pi*13*t_dense), label="13 Hz", alpha=0.7)
plt.stem(t_samples, x3_samples, linefmt="k-", markerfmt="ko", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Because `13 = 3 + fs`, the two sample phases differ by `2*pi*n`. This is the lecture's ambiguity made visible.

### Solution 7

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, sinc_reconstruct


def signal(t):
    return np.sin(2 * np.pi * 3 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, -1.0, 1.1)
t_dense = np.linspace(-1.0, 1.0, 2001)
x_reconstructed = sinc_reconstruct(t_samples, x_samples, t_dense)
x_at_samples = sinc_reconstruct(t_samples, x_samples, t_samples)
print(np.allclose(x_at_samples, x_samples, atol=1e-12))  # True

plt.plot(t_dense, signal(t_dense), label="original")
plt.plot(t_dense, x_reconstructed, "--", label="finite sinc")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

The supplied samples are interpolated exactly. Any visible edge error comes from truncating the theoretical infinite sinc sum to a finite record.

### Solution 8

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import sample_signal, zero_order_hold


def signal(t):
    return np.cos(2 * np.pi * 2 * t)


fs = 10.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0, endpoint=True)
t_dense = np.linspace(0.0, 1.0, 2001)
x_zoh = zero_order_hold(t_samples, x_samples, t_dense)

plt.plot(t_dense, signal(t_dense), label="original")
plt.step(t_dense, x_zoh, where="post", label="ZOH")
plt.stem(t_samples, x_samples, linefmt="C2-", markerfmt="C2o", basefmt="k-")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Each stored value is held until the next sample. The staircase is causal and practical but is not ideal reconstruction.

### Solution 9

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import zoh_frequency_response

T = 0.1
f = np.linspace(-20.0, 20.0, 4001)
H0 = zoh_frequency_response(f, T)

dc_gain = abs(zoh_frequency_response([0.0], T)[0])
first_null = 1 / T
print(dc_gain)    # 0.1
print(first_null) # 10.0 Hz

plt.plot(f, np.abs(H0))
plt.axvline(first_null, color="C1", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H0(f)|")
plt.grid(alpha=0.3)
plt.show()
```

The ZOH has DC gain `0.1` and its first positive magnitude null at `10 Hz`. Between DC and the null, the sinc envelope causes passband droop.

### Solution 10

```python
import numpy as np
import matplotlib.pyplot as plt
from sampling_skeleton import (
    magnitude_spectrum,
    meets_nyquist,
    sample_signal,
    sinc_reconstruct,
    zero_order_hold,
    zero_pad,
)


def signal(t):
    return np.sin(2*np.pi*4*t) + 0.4*np.cos(2*np.pi*11*t)


fs = 30.0
t_samples, x_samples = sample_signal(signal, fs, 0.0, 1.0)
print(meets_nyquist(fs, 11.0))  # True: 30 > 22

positive_times = t_samples[x_samples > 0]
positive_values = x_samples[x_samples > 0]
print("positive sample count:", positive_values.size)

x_padded = zero_pad(x_samples, right=4)
print("padded length:", x_padded.size)  # 34

f, amplitude = magnitude_spectrum(x_samples, fs)
t_dense = np.linspace(t_samples[0], t_samples[-1], 2001)
x_sinc = sinc_reconstruct(t_samples, x_samples, t_dense)
x_zoh = zero_order_hold(t_samples, x_samples, t_dense)

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(t_dense, signal(t_dense), label="original")
axes[0].plot(t_dense, x_sinc, "--", label="finite sinc")
axes[0].step(t_dense, x_zoh, where="post", label="ZOH")
axes[0].scatter(positive_times, positive_values, color="C3", label="positive samples")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].stem(f, amplitude)
axes[1].set(xlim=(0, fs/2), xlabel="Frequency (Hz)", ylabel="Amplitude")
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()
```

Both `4 Hz` and `11 Hz` lie below `fs/2 = 15 Hz`, so the strict sampling condition holds. Sinc aims at ideal band-limited interpolation; ZOH produces the expected staircase and high-frequency droop.
