import ctypes
import logging
import sys
import warnings

import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap, utils

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


# Infomap's own log with verbose=True (#87). Infomap 2.14 and older print it from C++;
# 2.15 and later send it through Python logging, so infostop attaches a handler to the
# "infomap" logger for the run and must leave the logger as it found it.

INFOMAP_LOGGER = logging.getLogger("infomap")


def _logger_state(logger):
    return (list(logger.handlers), logger.level, logger.propagate, logger.disabled)


@pytest.fixture
def infomap_logger():
    """The "infomap" logger, put back as it was after the test."""
    handlers, level, propagate, disabled = _logger_state(INFOMAP_LOGGER)
    yield INFOMAP_LOGGER
    INFOMAP_LOGGER.handlers[:] = handlers
    INFOMAP_LOGGER.setLevel(level)
    INFOMAP_LOGGER.propagate = propagate
    INFOMAP_LOGGER.disabled = disabled


def _list_handler():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    return handler, records


@pytest.mark.parametrize("model", [Infostop, SpatialInfomap])
def test_verbose_shows_infomap_log(capfd, infomap_logger, model):
    trace = _home_work_home()
    data = trace if model is Infostop else trace[:, :2]
    _flush_c_stdio()
    capfd.readouterr()

    model(verbose=True, num_trials=3).fit_predict(data)
    _flush_c_stdio()

    out = capfd.readouterr().out
    assert "codelength" in out
    assert "Trial 3/3" in out
    if utils._INFOMAP_LOGS_TO_PYTHON:
        # The log comes between infostop's "running Infomap..." and "done" lines
        lines = out.splitlines()
        start = lines.index("    ... running Infomap...")
        log = [i for i, line in enumerate(lines) if "codelength" in line]
        assert start < log[0] and log[-1] < lines.index("done")


@pytest.mark.parametrize("custom", [False, True])
def test_verbose_leaves_infomap_logger_as_it_was(infomap_logger, custom):
    if custom:
        infomap_logger.setLevel(logging.ERROR)
        infomap_logger.propagate = False
    before = _logger_state(infomap_logger)

    Infostop(verbose=True, num_trials=3).fit_predict(_home_work_home())
    SpatialInfomap(verbose=True, num_trials=3).fit_predict(_home_work_home()[:, :2])

    assert _logger_state(infomap_logger) == before


@pytest.mark.parametrize("step", ["create", "run"])
def test_verbose_restores_infomap_logger_when_infomap_raises(
    monkeypatch, infomap_logger, step
):
    before = _logger_state(infomap_logger)
    during = []

    class Failure(Exception):
        pass

    def fail(*args, **kwargs):
        during.append(_logger_state(infomap_logger))
        raise Failure

    if step == "create":
        monkeypatch.setattr(utils, "Infomap", fail)
    else:
        monkeypatch.setattr(utils.Infomap, "run", fail)
    with pytest.raises(Failure):
        Infostop(verbose=True).fit_predict(_home_work_home())

    assert _logger_state(infomap_logger) == before
    handlers, level, propagate, _ = during[0]
    if utils._INFOMAP_LOGS_TO_PYTHON:
        assert len(handlers) == 1 and level == logging.INFO and propagate is False
    else:
        assert during[0] == before


def test_verbose_false_attaches_no_handler(monkeypatch, infomap_logger):
    before = _logger_state(infomap_logger)
    during = []
    run = utils.Infomap.run

    def spy(self, *args, **kwargs):
        during.append(_logger_state(infomap_logger))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(utils.Infomap, "run", spy)
    Infostop(verbose=False).fit_predict(_home_work_home())

    assert during == [before]


def test_verbose_leaves_a_users_infomap_handler_alone(capfd, infomap_logger):
    # A user who attached a handler to the "infomap" logger gets the log there,
    # and infostop adds no second handler that would print it on stdout as well
    handler, records = _list_handler()
    infomap_logger.addHandler(handler)
    infomap_logger.setLevel(logging.INFO)
    before = _logger_state(infomap_logger)
    _flush_c_stdio()
    capfd.readouterr()

    Infostop(verbose=True, num_trials=3).fit_predict(_home_work_home())
    _flush_c_stdio()

    assert _logger_state(infomap_logger) == before
    if utils._INFOMAP_LOGS_TO_PYTHON:
        assert any("codelength" in record.getMessage() for record in records)
        assert "codelength" not in capfd.readouterr().out


def test_verbose_infomap_log_not_repeated_by_root_handler(infomap_logger):
    # As after logging.basicConfig(level=logging.INFO): the log is shown once, by
    # infostop's handler, and does not also reach the root logger's handlers
    root = logging.getLogger()
    handler, records = _list_handler()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        Infostop(verbose=True, num_trials=3).fit_predict(_home_work_home())
    finally:
        root.removeHandler(handler)
        root.setLevel(level)

    assert [record for record in records if record.name == "infomap"] == []
