"""Tests for the device transforms in pixell.curvedsky.

alm2map and map2alm send arrays that are not numpy arrays to ducc's CUDA
backend. The numpy tests run everywhere: they check that ring description
against the transforms curvedsky does itself. The JAX and CuPy tests need a
ducc0 built with the CUDA spherical harmonic backend, and are skipped
otherwise.
"""
import numpy as np
import pytest
import ducc0
from numpy.testing import assert_allclose
from pixell import curvedsky, enmap, lensing, utils

pmp = pytest.mark.parametrize

has_cuda = hasattr(ducc0.sht, "synthesis_cuda")
has_ffi  = hasattr(ducc0.sht, "sht_ffi_targets")


def jax_modules():
    """Import jax and build the transforms, or skip the test. Also skips when
    jax ends up on a different platform than the transforms are registered for,
    which happens when jax cannot use the GPU of the machine."""
    if not has_ffi:
        pytest.skip("this ducc0 build has no XLA handlers")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    # jax calls every accelerator "gpu", the handlers are registered for "CUDA"
    want = curvedsky._jax_platform().lower()
    if want in ("cuda", "rocm"): want = "gpu"
    if jax.default_backend() != want:
        pytest.skip("jax runs on the %s platform, the transforms are registered for %s"
            % (jax.default_backend(), curvedsky._jax_platform()))
    return jax, jax.numpy, curvedsky._jax_sht()


def geometry(kind="full", res=2.0, ydir=1, xdir=-1):
    """A CAR geometry with either direction of each axis: the full sky, a patch
    that covers part of every ring, or a declination band of the shape a Simons
    Observatory map has, which covers full rings over part of the sky"""
    if kind == "patch":
        shape, wcs = enmap.geometry(pos=np.deg2rad([[-30,40],[20,-60]]), res=np.deg2rad(res), proj="car")
    elif kind == "band":
        shape, wcs = enmap.band_geometry(np.deg2rad([-65,25]), res=np.deg2rad(res), proj="car")
    else:
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res), proj="car")
    wcs = wcs.deepcopy()
    for i, dir in [(0,xdir),(1,ydir)]:
        if (wcs.wcs.cdelt[i] > 0) != (dir > 0):
            wcs.wcs.cdelt[i] *= -1
            wcs.wcs.crpix[i]  = shape[1-i]+1-wcs.wcs.crpix[i]
    return shape, wcs


def random_alm(ainfo, ncomp, seed=1):
    rng = np.random.default_rng(seed)
    alm = rng.normal(size=(ncomp,ainfo.nelem)) + 1j*rng.normal(size=(ncomp,ainfo.nelem))
    alm[:,:ainfo.lmax+1] = alm[:,:ainfo.lmax+1].real
    return alm


def case(kind="full", ydir=1, xdir=-1, lmax=40, ncomp=3):
    """A geometry, its ring description, coefficients and the map curvedsky makes from them"""
    shape, wcs = geometry(kind=kind, ydir=ydir, xdir=xdir)
    ainfo = curvedsky.alm_info(lmax)
    alm   = random_alm(ainfo, ncomp)
    map   = curvedsky.alm2map(alm.copy(), enmap.zeros((ncomp,)+shape, wcs), spin=[0,2],
        ainfo=ainfo, method="cyl")
    return curvedsky.RingGeometry(shape, wcs), ainfo, alm, map


@pmp("kind", ["full", "patch", "band"])
@pmp("ydir", [1, -1])
@pmp("xdir", [1, -1])
def test_numpy_matches_curvedsky(kind, ydir, xdir):
    """The rings of the map itself give the same transforms as curvedsky's,
    which flips and pads the map first"""
    geom, ainfo, alm, map = case(kind=kind, ydir=ydir, xdir=xdir)
    assert geom.pixstride == (1 if xdir > 0 else -1)
    assert geom.padded == (kind == "patch")
    # curvedsky sends numpy arrays to its own code, so the device implementation
    # is called directly here; a geometry in the map slot reaches it either way
    out = curvedsky.alm2map(alm, geom, spin=[0,2], ainfo=ainfo)
    assert_allclose(out, np.asarray(map), atol=1e-12*np.abs(map).max())
    # analysis, with the quadrature weights of the geometry
    ref = curvedsky.map2alm(map, ainfo=ainfo, spin=[0,2], method="cyl")
    got = curvedsky._device_map2alm(map, spin=[0,2], ainfo=ainfo)
    assert_allclose(got, ref, atol=1e-12*np.abs(ref).max())
    # and as the plain adjoint of the synthesis, without weights
    ref = curvedsky.alm2map(None, map, spin=[0,2], method="cyl", adjoint=True, ainfo=ainfo)
    got = curvedsky._device_map2alm(map, spin=[0,2], ainfo=ainfo, weights=False)
    assert_allclose(got, ref, atol=1e-12*np.abs(ref).max())


@pmp("kind", ["numpy", "jax"])
def test_deriv_matches_curvedsky(kind):
    """The DERIV1 mode, which gives the two derivatives of a scalar field"""
    if kind == "jax": jax, jnp, jsht = jax_modules()
    shape, wcs = geometry()
    ainfo = curvedsky.alm_info(40)
    alm   = random_alm(ainfo, 1)[0]
    geom  = curvedsky.RingGeometry(shape, wcs)
    ref   = curvedsky.alm2map(alm, enmap.zeros((2,)+shape, wcs), deriv=True, ainfo=ainfo,
        method="cyl")
    a     = alm if kind == "numpy" else jnp.asarray(alm)
    got   = curvedsky.alm2map(a, geom, deriv=True, ainfo=ainfo)
    assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12*np.abs(ref).max())
    # and its adjoint, which reads the map and writes coefficients
    aref  = curvedsky.alm2map(np.zeros(ainfo.nelem, complex), ref, deriv=True, adjoint=True,
        ainfo=ainfo, method="cyl")
    m     = ref if kind == "numpy" else enmap.devmap(jnp.asarray(np.asarray(ref)), wcs)
    aout  = curvedsky._device_alm2map(None, m, deriv=True, adjoint=True, ainfo=ainfo) \
        if kind == "numpy" else curvedsky.alm2map(None, m, deriv=True, adjoint=True, ainfo=ainfo)
    assert_allclose(np.asarray(aout)[0], np.asarray(aref), atol=1e-12*np.abs(aref).max())


def test_map2alm_adjoint_matches_curvedsky():
    """map2alm with adjoint=True goes from coefficients to a weighted map"""
    geom, ainfo, alm, map = case()
    ref = curvedsky.map2alm_adjoint(alm, enmap.zeros(map.shape, map.wcs), ainfo=ainfo,
        spin=[0,2], method="cyl")
    got = curvedsky.map2alm((geom.shape, geom.wcs), alm, adjoint=True, ainfo=ainfo, spin=[0,2])
    assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12*np.abs(ref).max())


@pmp("niter", [1, 2])
def test_niter_matches_curvedsky(niter):
    geom, ainfo, alm, map = case(kind="patch")
    ref = curvedsky.map2alm(map, ainfo=ainfo, spin=[0,2], method="cyl", niter=niter)
    got = curvedsky._device_map2alm(map, spin=[0,2], ainfo=ainfo, niter=niter)
    assert_allclose(got, ref, atol=1e-12*np.abs(ref).max())


def test_devmap_carries_the_geometry():
    """A map on a device carries its wcs by being an enmap.devmap, so the
    transforms take it without being told the geometry"""
    jax, jnp, jsht = jax_modules()
    geom, ainfo, alm, map = case(ncomp=1)
    dev = curvedsky.alm2map(jnp.asarray(alm), geom, spin=[0], ainfo=ainfo)
    assert isinstance(dev, enmap.devmap)
    assert_allclose(np.asarray(dev.arr), np.asarray(map), atol=1e-12*np.abs(map).max())
    ref = curvedsky.map2alm(map, ainfo=ainfo, spin=[0], method="cyl")
    got = curvedsky.map2alm(dev, ainfo=ainfo, spin=[0])
    assert_allclose(np.asarray(got), ref, atol=1e-12*np.abs(ref).max())
    # and it survives a jit boundary, the pixels as the leaf, the geometry as
    # static data
    f = jax.jit(lambda d: curvedsky.map2alm(d, ainfo=ainfo, spin=[0]))
    assert_allclose(np.asarray(f(dev)), ref, atol=1e-12*np.abs(ref).max())


def test_geometry_from_tuple_and_output_array():
    shape, wcs = geometry()
    ainfo = curvedsky.alm_info(30)
    alm   = random_alm(ainfo, 1)
    ref   = curvedsky.alm2map(alm, (shape,wcs), spin=[0])
    assert ref.wcs is not None            # a numpy result comes back as an enmap
    out   = enmap.zeros((1,)+shape, wcs)
    got   = curvedsky.alm2map(alm, out, spin=[0])
    assert got is out
    assert_allclose(out, ref, atol=1e-13*np.abs(ref).max())


def test_roundtrip_is_exact():
    """On a grid with enough rings the weighted analysis inverts the synthesis"""
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5), proj="car")
    ainfo = curvedsky.alm_info(60)
    alm   = random_alm(ainfo, 1)
    geom  = curvedsky.RingGeometry(shape, wcs)
    map   = curvedsky.alm2map(alm, geom, spin=[0], ainfo=ainfo)
    got   = curvedsky.map2alm(map, spin=[0], ainfo=ainfo)
    assert_allclose(got, alm, atol=1e-10*np.abs(alm).max())


@pmp("kind", ["full", "patch", "band"])
@pmp("xdir", [1, -1])
def test_jax_matches_curvedsky(kind, xdir):
    jax, jnp, jsht = jax_modules()
    geom, ainfo, alm, map = case(kind=kind, xdir=xdir)
    out = curvedsky.alm2map(jnp.asarray(alm), geom, spin=[0,2], ainfo=ainfo)
    assert_allclose(np.asarray(out), np.asarray(map), atol=1e-12*np.abs(map).max())
    ref = curvedsky.map2alm(map, ainfo=ainfo, spin=[0,2], method="cyl")
    got = curvedsky.map2alm(enmap.devmap(jnp.asarray(np.asarray(map)), geom.wcs), spin=[0,2], ainfo=ainfo)
    assert_allclose(np.asarray(got), ref, atol=1e-12*np.abs(ref).max())


def test_jax_jit_and_vmap():
    jax, jnp, jsht = jax_modules()
    geom, ainfo, alm, map = case(ncomp=1)
    f   = jax.jit(lambda a: curvedsky.alm2map(a, geom, spin=[0], ainfo=ainfo))
    out = f(jnp.asarray(alm))
    assert_allclose(np.asarray(out[0]), np.asarray(map[0]), atol=1e-12*np.abs(map).max())
    batch = jnp.asarray(np.array([alm, 2*alm]))
    vout  = jax.vmap(f)(batch)
    assert_allclose(np.asarray(vout[0]), np.asarray(out), atol=1e-13*np.abs(map).max())
    assert_allclose(np.asarray(vout[1]), 2*np.asarray(out), atol=1e-13*np.abs(map).max())


def test_jax_gradient():
    """The gradient of a loss on the map agrees with the gradient through an
    explicit synthesis matrix, which fixes the convention for complex inputs"""
    jax, jnp, jsht = jax_modules()
    shape, wcs = geometry(res=15.0)
    ainfo = curvedsky.alm_info(4)
    geom  = curvedsky.RingGeometry(shape, wcs)
    alm   = random_alm(ainfo, 1)
    # columns of the real synthesis matrix, from unit coefficient responses
    cols = []
    for k in range(ainfo.nelem):
        for part in (1.0, 1j):
            a = np.zeros((1,ainfo.nelem), dtype=complex)
            a[0,k] = part
            if k <= ainfo.lmax and part == 1j: cols.append(np.zeros_like(cols[-1]))
            else: cols.append(np.asarray(curvedsky.alm2map(a, geom, spin=[0], ainfo=ainfo))[0].reshape(-1))
    Y = jnp.asarray(np.array(cols).T)
    w = jnp.asarray(np.random.default_rng(7).normal(size=Y.shape[0]))
    def loss(a):
        m = curvedsky.alm2map(a, geom, spin=[0], ainfo=ainfo).arr[0].reshape(-1)
        return jnp.sum(w*m**2)
    def loss_dense(a):
        v = jnp.stack([a[0].real, a[0].imag], axis=-1).reshape(-1)
        return jnp.sum(w*(Y @ v)**2)
    a = jnp.asarray(alm)
    assert_allclose(float(loss(a)), float(loss_dense(a)), rtol=1e-10)
    g1 = np.asarray(jax.grad(loss)(a))
    g2 = np.asarray(jax.grad(loss_dense)(a))
    assert_allclose(g1, g2, atol=1e-10*np.abs(g2).max())


@pmp("kind", ["full", "band"])
def test_cupy_matches_curvedsky(kind):
    cp = pytest.importorskip("cupy")
    if not has_cuda or ducc0.sht.cuda_emulated:
        pytest.skip("this ducc0 build has no CUDA transforms")
    geom, ainfo, alm, map = case(kind=kind)
    out = curvedsky.alm2map(cp.asarray(alm), geom, spin=[0,2], ainfo=ainfo)
    assert_allclose(cp.asnumpy(out), np.asarray(map), atol=1e-12*np.abs(map).max())
    ref = curvedsky.map2alm(map, ainfo=ainfo, spin=[0,2], method="cyl")
    got = curvedsky.map2alm(enmap.devmap(cp.asarray(np.asarray(map)), geom.wcs), spin=[0,2], ainfo=ainfo)
    assert_allclose(cp.asnumpy(got), ref, atol=1e-12*np.abs(ref).max())


@pmp("mode", ["STANDARD", "GRAD_ONLY"])
@pmp("spin", [0, 1, 2])
@pmp("lmax", [7, 24])
@pmp("dtype", [np.complex128, np.complex64])
def test_jax_raw_matches_ducc(mode, spin, lmax, dtype):
    """The flat array interface behind the device path, against ducc's own"""
    if mode != "STANDARD" and spin == 0: pytest.skip("GRAD_ONLY needs a nonzero spin")
    jax, jnp, jsht = jax_modules()
    nrings, nphi = lmax+1, 2*lmax+2
    rings = dict(theta=(np.arange(nrings)+0.5)*np.pi/nrings,
        nphi=np.full(nrings, nphi, np.uint64), phi0=np.zeros(nrings),
        ringstart=(np.arange(nrings)*nphi).astype(np.uint64))
    ainfo = curvedsky.alm_info(lmax)
    ncomp = 1 if spin == 0 or mode != "STANDARD" else 2
    alm   = random_alm(ainfo, ncomp).astype(dtype)
    alm[:,:spin] = 0
    tol   = 1e-12 if dtype == np.complex128 else 1e-5
    ref = ducc0.sht.synthesis(alm=alm, spin=spin, lmax=lmax, mode=mode, nthreads=1, **rings)
    out = jsht.synthesis(jnp.asarray(alm), spin=spin, lmax=lmax, mode=mode, **rings)
    assert_allclose(np.asarray(out), ref, atol=tol*np.abs(ref).max())
    aref = ducc0.sht.adjoint_synthesis(map=ref, spin=spin, lmax=lmax, mode=mode, nthreads=1, **rings)
    aout = jsht.adjoint_synthesis(jnp.asarray(ref), spin=spin, lmax=lmax, mode=mode, **rings)
    assert_allclose(np.asarray(aout), aref, atol=tol*np.abs(aref).max())


def test_jax_spin_list():
    """A list of spins does the same as separate transforms per spin group"""
    jax, jnp, jsht = jax_modules()
    geom, ainfo, alm, map = case()
    a   = jnp.asarray(alm)
    out = curvedsky.alm2map(a, geom, spin=[0,2], ainfo=ainfo)
    ref0 = curvedsky.alm2map(a[:1], geom, spin=[0], ainfo=ainfo)
    ref2 = curvedsky.alm2map(a[1:], geom, spin=[2], ainfo=ainfo)
    assert_allclose(np.asarray(out[:1]), np.asarray(ref0), atol=1e-13*np.abs(map).max())
    assert_allclose(np.asarray(out[1:]), np.asarray(ref2), atol=1e-13*np.abs(map).max())


def random_pos(npoints, seed=3):
    rng = np.random.default_rng(seed)
    pos = np.empty((2, npoints))
    pos[0] = np.arcsin(rng.uniform(-1., 1., npoints))
    pos[1] = rng.uniform(-np.pi, np.pi, npoints)
    return pos


# the component layouts alm2map_pos handles: a scalar field, a spin pair, a
# T, Q, U set, and the two derivatives of a scalar field
POS_CASES = [("spin0", [0], 1, False), ("spin1", [1], 2, False),
             ("spin2", [2], 2, False), ("tqu", [0,2], 3, False),
             ("deriv", [0,2], 1, True)]


@pmp("case", POS_CASES, ids=lambda c: c[0])
@pmp("dtype", [np.complex128, np.complex64])
def test_jax_pos_matches_curvedsky(case, dtype):
    """alm2map_pos on JAX arrays gives what it gives on numpy arrays"""
    jax, jnp, _ = jax_modules()
    _, spin, nalm, deriv = case
    ainfo = curvedsky.alm_info(40)
    pos   = random_pos(300)
    alm   = random_alm(ainfo, nalm).astype(dtype)
    if deriv: alm = alm[0]
    tol   = 1e-11 if dtype == np.complex128 else 1e-4
    kw    = dict(pos=pos, ainfo=ainfo, spin=spin, deriv=deriv)
    ref   = curvedsky.alm2map_pos(alm, **kw)
    out   = np.asarray(curvedsky.alm2map_pos(jnp.asarray(alm), **kw))
    assert_allclose(out, ref, atol=tol*np.abs(ref).max())

    sig  = np.random.default_rng(4).normal(size=ref.shape).astype(
        np.float64 if dtype == np.complex128 else np.float32)
    # each call gets its own copy: the numpy path used to negate a component of
    # the values it was given, and a shared array would have hidden that
    ref  = curvedsky.alm2map_pos(np.zeros_like(alm), map=sig.copy(), adjoint=True, **kw)
    out  = np.asarray(curvedsky.alm2map_pos(None, map=jnp.asarray(sig), adjoint=True, **kw))
    assert_allclose(out.reshape(ref.shape), ref, atol=tol*np.abs(ref).max())


def test_jax_pos_jit_and_epsilon():
    """The transform runs inside jit, and a looser epsilon still lands within
    the accuracy it asks for"""
    jax, jnp, _ = jax_modules()
    ainfo = curvedsky.alm_info(40)
    loc   = curvedsky._loc_from_pos(random_pos(200))
    alm   = random_alm(ainfo, 1)
    ref   = curvedsky.alm2map_pos(alm, loc=loc, ainfo=ainfo, spin=[0])
    f = jax.jit(lambda a: curvedsky.alm2map_pos(a, loc=loc, ainfo=ainfo, spin=[0]))
    assert_allclose(np.asarray(f(jnp.asarray(alm))), ref, atol=1e-11*np.abs(ref).max())
    out = curvedsky.alm2map_pos(jnp.asarray(alm), loc=loc, ainfo=ainfo, spin=[0],
                                epsilon=1e-5)
    assert_allclose(np.asarray(out), ref, atol=1e-4*np.abs(ref).max())


def test_jax_pos_gradient():
    """The gradient of a loss on the values at the positions agrees with the
    gradient through an explicit synthesis matrix, which fixes the convention
    for complex inputs"""
    jax, jnp, _ = jax_modules()
    ainfo = curvedsky.alm_info(4)
    loc   = curvedsky._loc_from_pos(random_pos(50))
    alm   = random_alm(ainfo, 1)
    kw    = dict(loc=loc, ainfo=ainfo, spin=[0])
    cols  = []
    for k in range(ainfo.nelem):
        for part in (1.0, 1j):
            a = np.zeros((1,ainfo.nelem), dtype=complex)
            a[0,k] = part
            if k <= ainfo.lmax and part == 1j: cols.append(np.zeros_like(cols[-1]))
            else: cols.append(np.asarray(curvedsky.alm2map_pos(a, **kw))[0])
    Y = jnp.asarray(np.array(cols).T)
    w = jnp.asarray(np.random.default_rng(7).normal(size=Y.shape[0]))
    def loss(a):
        return jnp.sum(w*curvedsky.alm2map_pos(a, **kw)[0]**2)
    def loss_dense(a):
        v = jnp.stack([a[0].real, a[0].imag], axis=-1).reshape(-1)
        return jnp.sum(w*(Y @ v)**2)
    a = jnp.asarray(alm)
    assert_allclose(float(loss(a)), float(loss_dense(a)), rtol=1e-10)
    g1 = np.asarray(jax.grad(loss)(a))
    g2 = np.asarray(jax.grad(loss_dense)(a))
    assert_allclose(g1, g2, atol=1e-10*np.abs(g2).max())


@pmp("case", POS_CASES, ids=lambda c: c[0])
def test_cupy_pos_matches_curvedsky(case):
    """alm2map_pos on CuPy arrays gives what it gives on numpy arrays"""
    cp = pytest.importorskip("cupy")
    if not has_cuda: pytest.skip("this ducc0 build has no CUDA backend")
    _, spin, nalm, deriv = case
    ainfo = curvedsky.alm_info(40)
    pos   = random_pos(300)
    alm   = random_alm(ainfo, nalm)
    if deriv: alm = alm[0]
    kw    = dict(pos=pos, ainfo=ainfo, spin=spin, deriv=deriv)
    ref   = curvedsky.alm2map_pos(alm, **kw)
    out   = cp.asnumpy(curvedsky.alm2map_pos(cp.asarray(alm), **kw))
    assert_allclose(out, ref, atol=1e-11*np.abs(ref).max())

    sig  = np.random.default_rng(4).normal(size=ref.shape)
    ref  = curvedsky.alm2map_pos(np.zeros_like(alm), map=sig.copy(), adjoint=True, **kw)
    out  = cp.asnumpy(curvedsky.alm2map_pos(cp.zeros_like(cp.asarray(alm)),
                                            map=cp.asarray(sig), adjoint=True, **kw))
    assert_allclose(out.reshape(ref.shape), ref, atol=1e-11*np.abs(ref).max())


@pmp("kind", ["cyl", "pos"])
def test_deriv_adjoint_is_a_transpose(kind):
    """The adjoint of a derivative transform is its transpose, and does not
    modify the values it is given. This runs on numpy alone: the device path
    and the numpy path only agree because both apply the sign of the
    declination derivative to the input of the adjoint."""
    ainfo = curvedsky.alm_info(20)
    alm   = random_alm(ainfo, 1)[0]
    if kind == "pos":
        pos = random_pos(40)
        kw  = dict(pos=pos, ainfo=ainfo, spin=[0,2], deriv=True)
        map = curvedsky.alm2map_pos(alm, **kw)
        adj = lambda m: curvedsky.alm2map_pos(np.zeros_like(alm), map=m, adjoint=True, **kw)
    else:
        # method cyl, which is the one the branch fixes; the 2d and healpix
        # paths still apply the sign to the output in both directions
        shape, wcs = geometry(res=10.0)
        kw  = dict(spin=[0,2], deriv=True, method="cyl")
        map = curvedsky.alm2map(alm, enmap.zeros((2,)+shape, wcs), **kw)
        adj = lambda m: curvedsky.alm2map(np.zeros_like(alm), map=m, adjoint=True, **kw)
    sig = np.random.default_rng(9).normal(size=map.shape)
    if kind == "cyl": sig = enmap.enmap(sig, map.wcs)
    kept = sig.copy()
    out = adj(sig)
    assert_allclose(sig, kept, atol=0)   # the adjoint leaves its input alone
    lmax = ainfo.lmax
    def almdot(a, b):
        return np.sum((a[:lmax+1].conj()*b[:lmax+1]).real) \
            + 2*np.sum((a[lmax+1:].conj()*b[lmax+1:]).real)
    assert_allclose(almdot(alm, out), np.sum(map*sig), rtol=1e-10)


def lensing_inputs(lmax=100, res=1.5, seed=5):
    """A geometry, a lensing potential and a T, E, B set to lens with"""
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res), proj="car")
    ainfo = curvedsky.alm_info(lmax)
    rng   = np.random.default_rng(seed)
    def draw(n):
        a = rng.normal(size=(n,ainfo.nelem)) + 1j*rng.normal(size=(n,ainfo.nelem))
        a[:,:lmax+1] = a[:,:lmax+1].real
        return a
    ell   = np.arange(lmax+1)
    scale = np.zeros(lmax+1)
    scale[2:] = 1.0/ell[2:]**2       # a red potential, so the deflections are smooth
    return (3,)+shape, wcs, ainfo.lmul(draw(1)*1e-1, scale)[0], draw(3)


@pmp("geodesic", [True, False])
def test_offset_by_grad_device_matches_numpy(geodesic):
    """The displaced positions are the same however the arrays are held"""
    rng  = np.random.default_rng(1)
    n    = 500
    ipos = np.stack([np.arcsin(rng.uniform(-0.99, 0.99, n)), rng.uniform(-np.pi, np.pi, n)])
    grad = rng.normal(size=(2,n))*1e-3
    grad[:,0] = 0    # zero deflections take their own branch of the trigonometry
    ref = lensing.offset_by_grad(ipos, grad, geodesic=geodesic, pol=True)
    out = lensing._offset_by_grad_device(ipos, grad, 3, geodesic)
    assert_allclose(out, ref, atol=1e-15)


@pmp("delta_theta", [None, np.deg2rad(20.0)])
def test_jax_lensing_matches_curvedsky(delta_theta):
    """A lensed map built from JAX coefficients is the one pixell builds from
    numpy ones, whether the work is done in one go or band by band. Note that
    delta_theta is in radians, so a value in degrees silently gives one band."""
    jax, jnp, _ = jax_modules()
    shape, wcs, phi_alm, cmb_alm = lensing_inputs()
    kw  = dict(output="lu", delta_theta=delta_theta)
    if delta_theta is not None:   # the band loop has to actually run
        assert int(shape[-2]/(shape[-2]//utils.nint(abs(delta_theta/utils.degree
            /wcs.wcs.cdelt[1]))+0.5)) < shape[-2]
    ref = lensing.lens_map_curved(shape, wcs, phi_alm, cmb_alm, **kw)
    out = lensing.lens_map_curved(shape, wcs, jnp.asarray(phi_alm), jnp.asarray(cmb_alm), **kw)
    for r, o in zip(ref, out):
        o = np.asarray(curvedsky.pixels(o))
        assert_allclose(o, r, atol=1e-9*np.abs(r).max())
    # lensing has to have moved the sky, or the comparison above is empty
    assert np.std(ref[0]-ref[1]) > 0.01*np.std(ref[1])


def test_jax_lensing_outputs():
    """The deflection and potential maps come back too, and carry their wcs"""
    jax, jnp, _ = jax_modules()
    shape, wcs, phi_alm, cmb_alm = lensing_inputs()
    kw  = dict(output="lap", delta_theta=None)
    ref = lensing.lens_map_curved(shape, wcs, phi_alm, cmb_alm, **kw)
    out = lensing.lens_map_curved(shape, wcs, jnp.asarray(phi_alm), jnp.asarray(cmb_alm), **kw)
    for r, o in zip(ref, out):
        assert o.wcs.wcs.compare(wcs.wcs)
        assert_allclose(np.asarray(curvedsky.pixels(o)), r, atol=1e-9*np.abs(r).max())


def test_position_gradient_is_nan_not_zero():
    """The derivative with respect to the positions is not implemented. It has
    to come back as nan: a zero would look like a valid answer, and the lensed
    map depends on the potential only through the positions, so a zero there
    would silently make a lensing gradient wrong rather than absent."""
    jax, jnp, _ = jax_modules()
    ainfo = curvedsky.alm_info(8)
    loc   = jnp.asarray(curvedsky._loc_from_pos(random_pos(30)))
    alm   = jnp.asarray(random_alm(ainfo, 1))
    w     = jnp.asarray(np.random.default_rng(3).normal(size=30))
    def loss(a, l):
        return jnp.sum(w*curvedsky.alm2map_pos(a, loc=l, ainfo=ainfo, spin=[0])[0]**2)
    ga = jax.grad(loss, argnums=0)(alm, loc)
    gl = jax.grad(loss, argnums=1)(alm, loc)
    assert np.all(np.isfinite(np.asarray(ga)))   # the supported one still works
    assert np.all(np.isnan(np.asarray(gl)))


def test_lensing_gradient_reports_the_missing_derivative():
    """A gradient of a lensed map with respect to the potential goes through
    the positions, so it is nan until that derivative exists; with respect to
    the unlensed CMB it is a real gradient."""
    jax, jnp, _ = jax_modules()
    shape, wcs, phi_alm, cmb_alm = lensing_inputs(lmax=24, res=6.0)
    w = jnp.asarray(np.random.default_rng(4).normal(size=tuple(shape[-2:])))
    def loss(p, c):
        m = lensing.lens_map_curved((1,)+tuple(shape[-2:]), wcs, p, c, spin=[0], output="l")[0]
        return jnp.sum(w*curvedsky.pixels(m)[0]**2)
    p, c = jnp.asarray(phi_alm), jnp.asarray(cmb_alm[:1])
    assert np.all(np.isfinite(np.asarray(jax.grad(loss, argnums=1)(p, c))))
    assert np.any(np.isnan(np.asarray(jax.grad(loss, argnums=0)(p, c))))


@pmp("dtype", [np.float64, np.float32])
def test_offset_by_grad_survives_a_single_precision_deflection(dtype):
    """A deflection field in single precision must still give finite positions.
    cos(d) of a small single precision deflection rounds to one, which drives
    the cosine of the displaced colatitude to the end of its range near the
    poles; the trigonometry runs in the precision of the positions so that it
    cannot."""
    lmax = 200
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(20.0/60), proj="car")
    ainfo = curvedsky.alm_info(lmax)
    rng   = np.random.default_rng(1)
    a     = rng.normal(size=(1,ainfo.nelem)) + 1j*rng.normal(size=(1,ainfo.nelem))
    a[:,:lmax+1] = a[:,:lmax+1].real
    ell   = np.arange(lmax+1)
    scale = np.zeros(lmax+1)
    scale[2:] = 1e-3/ell[2:]**2        # a few arcmin of deflection, as the real sky has
    phi   = ainfo.lmul(a, scale)[0]
    grad  = curvedsky.alm2map(phi, enmap.zeros((2,)+shape, wcs, dtype), deriv=True)
    out   = lensing.offset_by_grad(enmap.posmap(shape, wcs), grad, pol=True)
    assert np.all(np.isfinite(out))


@pmp("geodesic", [True, False])
def test_offset_by_grad_zero_deflection_at_a_pole(geodesic):
    """A point with no deflection stays where it is and is not rotated, at a
    pole as anywhere else. Nudging a zero deflection to something tiny, as this
    used to, makes cos(d) round to one, so at a pole the sine of the displaced
    colatitude cancels to zero and the longitude divides by it."""
    rng  = np.random.default_rng(1)
    n    = 500
    ipos = np.stack([np.arcsin(rng.uniform(-0.999, 0.999, n)), rng.uniform(-np.pi, np.pi, n)])
    grad = rng.normal(size=(2,n))*1e-3
    ipos[0,0], grad[:,0] = np.pi/2, 0     # north pole, no deflection
    ipos[0,1], grad[:,1] = -np.pi/2, 0    # south pole, no deflection
    grad[:,2] = 0                         # no deflection away from a pole
    ipos[0,3] = np.pi/2                   # a pole with a real deflection
    out = lensing.offset_by_grad(ipos, grad, geodesic=geodesic, pol=True)
    assert np.all(np.isfinite(out))
    for i in (0, 1, 2):
        assert_allclose(out[:2,i], ipos[:,i], atol=0)
        assert_allclose(out[2,i], 0.0, atol=0)
    # the device path shares the trigonometry and must agree exactly
    assert_allclose(lensing._offset_by_grad_device(ipos, grad, 3, geodesic), out, atol=0)
