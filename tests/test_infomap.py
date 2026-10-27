import ctypes
import sys
import warnings

import numpy as np

from infostop import Infostop, SpatialInfomap

HOME = (55.70, 12.50)
WORK = (55.71, 12.52)


def _stay(rng, lat, lon, t0, n=30):
    """`n` points, one per minute, scattered a few meters around (lat, lon)."""
    return np.column_stack(
        [
            lat + rng.normal(0, 2e-5, n),
            lon + rng.normal(0, 2e-5, n),
            t0 + 60 * np.arange(n),
        ]
    )


def _move(a, b, t0, n=10):
    """`n` points, one per minute, on the straight line from `a` to `b`."""
    f = np.linspace(0, 1, n + 2)[1:-1]
    return np.column_stack(
        [
            a[0] + f * (b[0] - a[0]),
            a[1] + f * (b[1] - a[1]),
            t0 + 60 * np.arange(n),
        ]
    )


def _home_work_home(seed=0):
    """Home, work, home stays with moving segments in between (rows 30-39, 70-79)."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            _stay(rng, *HOME, 0),
            _move(HOME, WORK, 1800),
            _stay(rng, *WORK, 2400),
            _move(WORK, HOME, 4200),
            _stay(rng, *HOME, 4800),
        ]
    )


def _flush_c_stdio():
    """Infomap prints from C++, which libc buffers when stdout is not a terminal.
    Flush it so `capfd` sees everything printed so far."""
    if sys.platform != "win32":
        ctypes.CDLL(None).fflush(None)


def test_fit_predict_runs_on_installed_infomap():
    trace = _home_work_home()
    labels = Infostop().fit_predict(trace)

    assert labels.shape == (len(trace),)
    home1, move1, work, move2, home2 = np.split(labels, [30, 40, 70, 80])
    assert np.all(move1 == -1) and np.all(move2 == -1)
    assert len(set(home1)) == 1 and len(set(work)) == 1
    assert home1[0] != -1 and work[0] != -1
    assert home1[0] == home2[0] != work[0]
    assert np.all(home2 == home1[0])


def test_spatial_infomap_runs_on_installed_infomap():
    rng = np.random.default_rng(1)
    points = np.vstack([_stay(rng, *HOME, 0), _stay(rng, *WORK, 0)])[:, :2]
    labels = SpatialInfomap().fit_predict(points)

    assert labels.shape == (len(points),)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_no_deprecated_infomap_api():
    trace = _home_work_home()
    with warnings.catch_warnings():
        # Deprecation warnings that are attributed to infostop's own code, e.g.
        # Infomap's warning for the `Infomap.modules` accessor (pending in 2.15)
        for category in (DeprecationWarning, PendingDeprecationWarning):
            warnings.filterwarnings("error", category=category, module="infostop")
        Infostop().fit_predict(trace)
        Infostop(weighted=True).fit_predict(trace)
        SpatialInfomap().fit_predict(trace[:, :2])


def test_verbose_false_prints_nothing(capfd):
    trace = _home_work_home()
    _flush_c_stdio()
    capfd.readouterr()

    Infostop(verbose=False).fit_predict(trace)
    SpatialInfomap(verbose=False).fit_predict(trace[:, :2])
    _flush_c_stdio()

    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""
