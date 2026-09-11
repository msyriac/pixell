"""A quick benchmark of the CPU and GPU transforms, to run after installing pixell with a CUDA
build of ducc0:

    pytest tests/test_device_benchmark.py

It times curvedsky.alm2map and curvedsky.map2alm for a T, Q, U set (spins 0 and 2) on a full
sky CAR map, first with numpy arrays on the CPU, then with CuPy and JAX arrays on the GPU,
checks that every GPU result agrees with the CPU one, and prints a table of the timings. The
band limit is PIXELL_BENCH_LMAX (default 1000), with two pixels per wavelength at that band
limit. A path whose library is not installed is skipped. With an emulated ducc0 build the band
limit is capped at 64, since the timings then only exercise the code.
"""

import os
import shutil
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # let jax share the GPU with ducc

import ducc0
import numpy as np
import pytest
from pixell import curvedsky, enmap
from test_curvedsky_device import has_cuda, jax_modules, random_alm

LMAX = int(os.environ.get("PIXELL_BENCH_LMAX", 1000))
NREP = 3
SPIN = [0, 2]


def best_time(fn, args=lambda: ()):
    """Calls fn once to warm up (plans, compilation), then returns its output and the fastest of
    NREP timed calls. args makes fresh inputs for each call, outside the timed region."""
    out = fn(*args())
    times = []
    for _ in range(NREP):
        a = args()
        t0 = time.perf_counter()
        out = fn(*a)
        times.append(time.perf_counter() - t0)
    return out, min(times)


def rel_diff(a, ref):
    return float(np.max(np.abs(a - ref)) / np.max(np.abs(ref)))


def gpu_name():
    if not shutil.which("nvidia-smi"):
        return "unknown GPU"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    )
    return out.stdout.splitlines()[0].strip() if out.stdout.strip() else "unknown GPU"


def test_cpu_and_gpu_transforms(capsys):
    """Times the CPU and GPU transforms and checks that they agree"""
    if not has_cuda:
        pytest.skip("this ducc0 build has no CUDA backend")
    emulated = ducc0.sht.cuda_emulated
    lmax = min(LMAX, 64) if emulated else LMAX
    shape, wcs = enmap.fullsky_geometry(res=np.pi / (2 * lmax), proj="car")
    ainfo = curvedsky.alm_info(lmax)
    geom = curvedsky.RingGeometry(shape, wcs)
    alm = random_alm(ainfo, 3)

    # the CPU reference, with numpy arrays
    omap = enmap.zeros((3,) + shape, wcs)
    cpu_map, cpu_syn = best_time(
        lambda a: np.array(curvedsky.alm2map(a, omap, spin=SPIN, ainfo=ainfo, method="cyl")),
        lambda: (alm.copy(),),
    )
    cpu_alm, cpu_ana = best_time(
        lambda m: curvedsky.map2alm(m, ainfo=ainfo, spin=SPIN, method="cyl"),
        lambda: (enmap.ndmap(cpu_map.copy(), wcs),),
    )
    rows, skipped = [("CPU (numpy)", cpu_syn, cpu_ana, 0.0, 0.0)], []

    try:
        import cupy as cp
    except ImportError:
        cp = None
        skipped.append("CuPy: not installed")
    if cp is not None and emulated:
        skipped.append("CuPy: needs the real CUDA build of ducc0")
    elif cp is not None:
        sync = cp.cuda.Device().synchronize
        calm, cmap = cp.asarray(alm), enmap.devmap(cp.asarray(cpu_map), wcs)

        def cupy_run(fn, x):
            out = fn(x)
            sync()
            return cp.asnumpy(getattr(out, "arr", out))

        m, t_syn = best_time(
            lambda: cupy_run(lambda a: curvedsky.alm2map(a, geom, spin=SPIN, ainfo=ainfo), calm)
        )
        a, t_ana = best_time(
            lambda: cupy_run(lambda x: curvedsky.map2alm(x, spin=SPIN, ainfo=ainfo), cmap)
        )
        rows.append(("CuPy", t_syn, t_ana, rel_diff(m, cpu_map), rel_diff(a, cpu_alm)))

    try:
        jax, jnp, _ = jax_modules()
    except pytest.skip.Exception as e:
        jax = None
        skipped.append(f"JAX: {e.msg}")
    if jax is not None:
        jalm, jmap = jnp.asarray(alm), jnp.asarray(cpu_map)
        syn = jax.jit(lambda a: curvedsky.alm2map(a, geom, spin=SPIN, ainfo=ainfo).arr)
        ana = jax.jit(lambda x: curvedsky.map2alm(enmap.devmap(x, wcs), spin=SPIN, ainfo=ainfo))

        def jax_run(fn, x):
            out = fn(x)
            float(jnp.real(out.ravel()[0]))  # waits for the device to finish
            return out

        m, t_syn = best_time(lambda: jax_run(syn, jalm))
        a, t_ana = best_time(lambda: jax_run(ana, jmap))
        rows.append(
            (
                "JAX (jit)",
                t_syn,
                t_ana,
                rel_diff(np.asarray(m), cpu_map),
                rel_diff(np.asarray(a), cpu_alm),
            )
        )

    if len(rows) == 1:
        pytest.skip("no GPU path available: " + "; ".join(skipped))
    device = "emulated CUDA backend on the CPU, timings not meaningful" if emulated else gpu_name()
    lines = [
        f"pixell CPU and GPU transforms: lmax {lmax}, full sky CAR {shape[-2]}x{shape[-1]}, spins {SPIN}, float64",
        f"CPU threads: {os.environ.get('OMP_NUM_THREADS', f'all ({os.cpu_count()})')}; GPU: {device}",
        f"{'':<13}{'alm2map s':>11}{'map2alm s':>11}{'alm2map speedup':>17}{'map2alm speedup':>17}"
        f"{'alm2map diff':>14}{'map2alm diff':>14}",
    ]
    for name, t_syn, t_ana, d_syn, d_ana in rows:
        diff = (
            f"{'reference':>14}{'':>14}"
            if name.startswith("CPU")
            else f"{d_syn:14.1e}{d_ana:14.1e}"
        )
        lines.append(
            f"{name:<13}{t_syn:11.4f}{t_ana:11.4f}{cpu_syn / t_syn:16.1f}x{cpu_ana / t_ana:16.1f}x{diff}"
        )
    lines += [f"skipped: {s}" for s in skipped]
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    for name, _, _, d_syn, d_ana in rows[1:]:
        assert max(d_syn, d_ana) < 1e-10, f"{name} differs from the CPU transforms"
