"""
image_conv.py  --  TASK B: blurring an image through the frequency domain.

YOUR CODE GOES HERE. image_utils.py (loading, saving, kernels, comparison
figures) and bench_utils.py (timing, runtime plots) are provided; the
transform, the padding logic and the convolution are yours.

Usage (the command line is already wired up for you):

    python3 image_conv.py images/skyline512.png --kernel bokeh --param 9 \
        --engine fft --out-dir outputs/skyline_bokeh
    python3 image_conv.py images/sunset512.png --gray --kernel motion \
        --param 41 --engine fft --out-dir outputs/sunset_motion
    python3 image_conv.py images/skyline512.png --benchmark \
        --out-dir outputs/benchmark

Restrictions: no numpy.fft / scipy.fft / scipy.ndimage / cv2 / PIL filters,
no numpy.convolve, no scipy.signal. Every transform goes through your own
transforms.py.
"""

import argparse
import os

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from image_utils import (load_image, make_kernel, save_comparison, save_image,
                         save_kernel_preview)
from io_utils import write_report
from transforms import ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer, next_power_of_two


def transform_2d(plane, engine):
    """
    2D forward transform of a single 2D array, by separability.

    The 2D DFT factorises into 1D transforms: transform every ROW, then
    transform every COLUMN of the result (the order does not matter). This is
    the only reason a 2D transform is affordable -- doing it directly from the
    2D definition would be O(N^4).

    Parameters
    ----------
    plane : 2D array_like, shape (P, Q)
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of complex128, shape (P, Q)
    """
    # TODO: implement this function
    P,Q=plane.shape
    X_temp = np.zeros((P, Q),dtype='complex')
    X=X_temp.copy()

    for p in range(P):
        X_temp[p, :]=engine.transform(plane[p, :])

    for q in range(Q):
        X[:, q]=engine.transform(X_temp[:, q])

    return X
def inverse_2d(spectrum, engine):
    """
    2D inverse transform, the same way round. Shape is preserved.
    """
    # TODO: implement this function
    P,Q=spectrum.shape
    X_temp = np.zeros((P, Q),dtype='complex')
    X=X_temp.copy()

    for p in range(P):
        X_temp[p, :]=engine.inverse(spectrum[p, :])

    for q in range(Q):
        X[:, q]=engine.inverse(X_temp[:, q])

    return X

def convolve_plane(plane, kernel, engine, circular=False):
    """
    Convolve one 2D plane with ``kernel`` through the frequency domain.

    Returns an array the SAME size as the input plane.

    circular=False (the normal case -- linear convolution):
        The full convolution of an (H, W) image with a (kh, kw) kernel is
        (H+kh-1, W+kw-1). Zero-pad both arrays to at least that size before
        transforming -- with FFTTransformer, pad further, up to a power of two
        in each dimension (every engine exposes a ``name`` attribute, so
        ``engine.name == "fft"`` tells you which rule applies). Multiply the
        two spectra, inverse-transform, take the real part, then crop the
        (H, W) window that corresponds to the original pixels: rows
        kh//2 .. kh//2+H-1 and columns kw//2 .. kw//2+W-1
        (the kernel sits at the origin of the padded array, so the result is
        offset by half the kernel -- forget this and your image comes out
        shifted diagonally).

    circular=True (the deliberate mistake -- see the specification):
        Transform at exactly (H, W) with no padding at all, with the kernel
        wrapped around the origin (np.roll is allowed -- it is not a
        transform). The output is the CIRCULAR convolution: content that
        should have fallen off one edge reappears on the opposite edge. The
        provided images are 256x256 and 512x512, so this path works with the
        radix-2 FFT directly.

    Parameters
    ----------
    plane : 2D numpy array of float, values in [0, 1]
    kernel : 2D numpy array of float, sums to 1
    engine : DFTAnalyzer or FFTTransformer
    circular : bool

    Returns
    -------
    numpy.ndarray of float64, same shape as ``plane``
    """
    # TODO: implement this function
    H,W=plane.shape
    kh,kw=kernel.shape

    if circular:
        P,Q=H,W
        plane_padded=plane

        kernel_padded=np.zeros((P,Q))
        kernel_padded[:kh,:kw]=kernel
        kernel_padded = np.roll(kernel_padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    else:
        full_h=H+kh-1
        full_w=W+kw-1
        if engine.name=="fft":
            P,Q=next_power_of_two(full_h),next_power_of_two(full_w)
        else:
            P,Q=full_h,full_w
        plane_padded=np.zeros((P,Q))
        plane_padded[:H,:W]=plane

        kernel_padded=np.zeros((P,Q))
        kernel_padded[:kh,:kw]=kernel

    spectrum_plane=transform_2d(plane_padded,engine)
    spectrum_kernel=transform_2d(kernel_padded,engine)
    conv=inverse_2d(spectrum_plane*spectrum_kernel,engine).real

    if circular:
        return conv
    r0,c0= kh//2,kw//2
    return conv[r0:r0+H,c0:c0+W]
    

def convolve_image(image, kernel, engine, circular=False):
    """
    Apply convolve_plane to a whole image.

    A grayscale image is (H, W); a colour image is (H, W, 3) and each colour
    plane is convolved independently, then stacked back together.
    """
    # TODO: implement this function
    if image.ndim==2:
        return convolve_plane(image, kernel, engine, circular)
    else:
        planes=[]
        for c in range(3):
            planes.append(convolve_plane(image[:, :, c], kernel, engine, circular))
        return np.stack(planes,axis=-1)

def convolve_plane_direct(plane, kernel):
    """
    Spatial convolution, written out literally, as the correctness oracle and
    the third benchmark curve.

        out[r, c] = sum_i sum_j  plane[r + kh//2 - i, c + kw//2 - j] * kernel[i, j]

    with out-of-range pixels treated as zero. Four nested loops, O(N^2 K^2),
    no NumPy vectorisation -- this one is meant to be slow and obviously
    correct. It is never applied to a full 512x512 image (see run_single).
    """
    # TODO: implement this function
    H,W=plane.shape
    kh,kw=kernel.shape
    # plain Python floats: numpy *scalar* indexing costs ~5x per element and
    # this loop runs H*W*kh*kw times. Same four loops, same arithmetic order.
    p_rows=plane.tolist()
    k_rows=kernel.tolist()
    out=np.zeros((H,W))
    out_rows=out.tolist()
    for r in range(H):
        for c in range(W):
            acc=0.0
            for i in range(kh):
                pr = r + kh//2 - i
                if not (0 <= pr < H):
                    continue
                p_row=p_rows[pr]
                k_row=k_rows[i]
                for j in range(kw):
                    pc = c + kw//2 - j
                    if 0 <= pc < W:
                        acc += p_row[pc] * k_row[j]
            out_rows[r][c]=acc

    return np.array(out_rows)
def run_single(path, kernel_name, param, engine_name, out_dir, gray=False):
    """
    Blur one image and write the required outputs.

    Build the kernel with image_utils.make_kernel:
        bokeh    -> make_kernel("bokeh", radius=param)
        gaussian -> make_kernel("gaussian", size=param)
        box      -> make_kernel("box", size=param)
        motion   -> make_kernel("motion", length=param, angle=30.0)

    Must produce, inside ``out_dir``:
      blurred.png     -- the linear (zero-padded) convolution
      wraparound.png  -- the same blur computed circularly, with no padding
      kernel.png      -- image_utils.save_kernel_preview of the kernel
      comparison.png  -- image_utils.save_comparison of original / blurred /
                         wraparound, side by side
      report.txt      -- image path and size, kernel name and size, engine,
                         the linear-convolution size, the transform size you
                         actually used, and the verification result. It is
                         written by your code; there is no separate write-up
                         to hand in.

    Verification: convolve the top-left 64x64 corner of the image (first colour
    plane, if colour) both ways -- convolve_plane and convolve_plane_direct --
    and report max |spectral - direct|. It should be ~1e-15, and anything above
    1e-9 is a bug, not rounding.
    """
    image = load_image(path, as_gray=gray)

    if kernel_name == "bokeh":
        kernel = make_kernel("bokeh", radius=param)
    elif kernel_name == "gaussian":
        kernel = make_kernel("gaussian", size=param)
    elif kernel_name == "box":
        kernel = make_kernel("box", size=param)
    elif kernel_name == "motion":
        kernel = make_kernel("motion", length=param, angle=30.0)
    else:
        raise ValueError("unknown kernel: %r" % kernel_name)

    engines = {"dft": DFTAnalyzer, "fft": FFTTransformer, "arbitrary": ArbitraryLengthFFT}
    engine = engines[engine_name]()

    blurred = convolve_image(image, kernel, engine, circular=False)
    wraparound = convolve_image(image, kernel, engine, circular=True)

    save_image(blurred, os.path.join(out_dir, "blurred.png"))
    save_image(wraparound, os.path.join(out_dir, "wraparound.png"))
    save_kernel_preview(kernel, os.path.join(out_dir, "kernel.png"),
                        title="%s kernel" % kernel_name)
    save_comparison([image, blurred, wraparound],
                    ["original", "linear convolution (zero-padded)",
                     "circular convolution (no padding)"],
                    os.path.join(out_dir, "comparison.png"),
                    suptitle="%s, %s kernel %dx%d, engine=%s"
                    % (os.path.basename(path), kernel_name,
                       kernel.shape[0], kernel.shape[1], engine_name))

    # verification: top-left 64x64 corner, first colour plane if colour
    plane = image[:64, :64] if image.ndim == 2 else image[:64, :64, 0]
    spectral = convolve_plane(plane, kernel, engine, circular=False)
    direct = convolve_plane_direct(plane, kernel)
    max_diff = float(np.max(np.abs(spectral - direct)))
    verdict = "MATCH" if max_diff < 1e-9 else "MISMATCH"

    kh, kw = kernel.shape
    full_h, full_w = image.shape[0] + kh - 1, image.shape[1] + kw - 1
    if engine.name == "fft":
        N_h, N_w = next_power_of_two(full_h), next_power_of_two(full_w)
    else:
        N_h, N_w = full_h, full_w

    def field(label, value):
        return "%-19s : %s" % (label, value)

    write_report(os.path.join(out_dir, "report.txt"), [
        "Task B -- 2D convolution through the frequency domain",
        field("image", "%s  (%d x %d, %s)"
              % (path, image.shape[0], image.shape[1],
                 "gray" if image.ndim == 2 else "RGB")),
        field("kernel", "%s  (%d x %d)" % (kernel_name, kh, kw)),
        field("engine", engine_name),
        field("linear-conv size", "%d x %d" % (full_h, full_w)),
        field("transform size", "%d x %d" % (N_h, N_w)),
        field("max |spectral - direct| on 64x64 crop", "%.3e" % max_diff),
        field("verification", verdict),
    ])

    print(verdict, "max diff", max_diff)
    if verdict == "MISMATCH":
        raise AssertionError("convolve_plane vs convolve_plane_direct mismatch")

    return blurred, wraparound


# ---------------------------------------------------------------------------
# PROVIDED -- run_benchmark is already written. It calls your convolve_plane
# and convolve_plane_direct, so it starts working as soon as those are
# correct. You do not need to modify anything below (though you may extend
# it).
# ---------------------------------------------------------------------------
IMAGE_SIZES = [16, 32, 64, 128, 256, 512]
KERNEL_RADII = [1, 3, 7, 15, 31]
BENCH_RADIUS = 7            # kernel used for the growing-image study
BENCH_SIZE = 256            # image crop used for the growing-kernel study
TIME_BUDGET = 20.0          # stop a sweep once one measurement exceeds this
                            # (raised from 8.0: this machine is ~2.4x slower than
                            #  the reference, so 8.0 truncated the K=63 point)


def run_benchmark(path, out_dir):
    """
    Two timing studies, two plots, both on one grayscale plane:

      1. growing image, fixed kernel   -> runtime_vs_image_size.png
      2. growing kernel, fixed image   -> runtime_vs_kernel_size.png

    plus both timing tables in report.txt. Each sweep stops early once a
    single measurement exceeds TIME_BUDGET seconds, so a slow machine simply
    produces a shorter curve rather than hanging.
    """
    full = load_image(path, as_gray=True)

    def sweep(label, make_call, points):
        """points: list of (x_value, zero-argument-callable-factory input)."""
        xs, ys = [], []
        print("%s:" % label)
        for x, arg in points:
            seconds = time_best(make_call(arg), repeats=1)
            xs.append(x)
            ys.append(seconds)
            print("  %8s   %9.4f s" % (x, seconds))
            if seconds > TIME_BUDGET:
                print("  (stopping this curve -- over the time budget)")
                break
        return xs, ys

    # ---- study 1: fixed kernel, growing image
    kernel = make_kernel("bokeh", radius=BENCH_RADIUS)
    crops = [(n, full[:n, :n].copy()) for n in IMAGE_SIZES]

    size_series = {}
    size_series["Naive DFT (row-column)"] = sweep(
        "naive DFT", lambda img: (lambda: convolve_plane(img, kernel, DFTAnalyzer())), crops)
    size_series["Radix-2 FFT (row-column)"] = sweep(
        "radix-2 FFT", lambda img: (lambda: convolve_plane(img, kernel, FFTTransformer())), crops)
    size_series["Direct spatial convolution"] = sweep(
        "direct spatial", lambda img: (lambda: convolve_plane_direct(img, kernel)), crops)

    size_plot = os.path.join(out_dir, "runtime_vs_image_size.png")
    plot_runtime_curve(size_series, size_plot,
                       title="Task B: %d x %d blur of an N x N image" % kernel.shape,
                       xlabel="image side length N (pixels)",
                       references=("n3", "n2"))

    # ---- study 2: fixed image, growing kernel
    image = full[:BENCH_SIZE, :BENCH_SIZE].copy()
    kernels = [(make_kernel("bokeh", radius=r).shape[0], make_kernel("bokeh", radius=r))
               for r in KERNEL_RADII]

    kernel_series = {}
    kernel_series["Direct spatial convolution"] = sweep(
        "direct spatial", lambda k: (lambda: convolve_plane_direct(image, k)), kernels)
    kernel_series["Radix-2 FFT (row-column)"] = sweep(
        "radix-2 FFT", lambda k: (lambda: convolve_plane(image, k, FFTTransformer())), kernels)

    kernel_plot = os.path.join(out_dir, "runtime_vs_kernel_size.png")
    plot_runtime_curve(kernel_series, kernel_plot,
                       title="Task B: %d x %d image, growing kernel" % image.shape,
                       xlabel="kernel side length K (pixels)",
                       references=("n2",))

    write_report(os.path.join(out_dir, "report.txt"),
                 ["Task B -- runtime benchmark", "",
                  "Study 1: fixed %d x %d kernel, growing image" % kernel.shape, ""]
                 + timing_table_lines(size_series, size_label="N")
                 + ["", "plot: %s" % os.path.basename(size_plot), "",
                    "Study 2: fixed %d x %d image, growing kernel" % image.shape, ""]
                 + timing_table_lines(kernel_series, size_label="K")
                 + ["", "plot: %s" % os.path.basename(kernel_plot)])
    print("wrote", size_plot, "and", kernel_plot)


def main():
    ap = argparse.ArgumentParser(description="2D convolution by DFT/FFT")
    ap.add_argument("image", help="path to the input image")
    ap.add_argument("--kernel", default="bokeh",
                    choices=["bokeh", "gaussian", "box", "motion"])
    ap.add_argument("--param", type=float, default=9,
                    help="bokeh radius / gaussian size / box size / motion length")
    ap.add_argument("--engine", default="fft", choices=["dft", "fft", "arbitrary"])
    ap.add_argument("--gray", action="store_true", help="process as grayscale")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the timing study instead of a single blur")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.benchmark:
        run_benchmark(args.image, args.out_dir)
    else:
        run_single(args.image, args.kernel, args.param, args.engine,
                   args.out_dir, gray=args.gray)


if __name__ == "__main__":
    main()
