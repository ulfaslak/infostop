import types

import numpy as np
import pytest

import infostop.models
from infostop import Infostop


def make_trace(seed, places, n_per_visit):
    """Gaussian blobs of points at each of `places`, joined by moving segments
    (10 points along a straight line), one point per 60 s."""
    rng = np.random.default_rng(seed)
    parts, t = [], 0
    for i, ((lat, lon), n) in enumerate(zip(places, n_per_visit)):
        if i > 0:
            a, b = np.array(places[i - 1]), np.array([lat, lon])
            f = np.linspace(0, 1, 12)[1:-1, None]
            parts.append(np.column_stack([a + f * (b - a), t + 60 * np.arange(10)]))
            t += 600
        parts.append(
            np.column_stack(
                [
                    lat + rng.normal(0, 2e-5, n),
                    lon + rng.normal(0, 2e-5, n),
                    t + 60 * np.arange(n),
                ]
            )
        )
        t += 60 * n + 60
    return np.vstack(parts)


A, B, C = (55.70, 12.50), (55.71, 12.52), (55.70, 12.53)
TRACE_1 = make_trace(0, [A, B, A, C], [30, 20, 25, 15])
TRACE_2 = make_trace(1, [B, C, B], [20, 30, 10])
METRICS = {
    "haversine": lambda t: t,
    "euclidean": lambda t: t * [1e5, 1e5, 1][: t.shape[1]],  # degrees to ~metres
}


@pytest.fixture
def stationary_event_calls(monkeypatch):
    """Record the number of rows of every trace passed to the C++ step."""
    calls = []
    cpputils = infostop.models.cpputils

    def spy(coords, *args):
        calls.append(len(coords))
        return cpputils.get_stationary_events(coords, *args)

    monkeypatch.setattr(
        infostop.models,
        "cpputils",
        types.SimpleNamespace(get_stationary_events=spy),
    )
    return calls


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("n_columns", [2, 3])
def test_empty_user_gets_empty_labels(metric, n_columns, stationary_event_calls):
    trace_1, trace_2 = (METRICS[metric](t[:, :n_columns]) for t in (TRACE_1, TRACE_2))
    empty = np.empty((0, n_columns))
    model = Infostop(distance_metric=metric)

    labels = model.fit_predict([empty, trace_1, empty, trace_2, empty])
    expected = Infostop(distance_metric=metric).fit_predict([trace_1, trace_2])

    assert [len(x) for x in labels] == [0, len(trace_1), 0, len(trace_2), 0]
    for i in (0, 2, 4):
        assert labels[i].dtype == labels[1].dtype
    np.testing.assert_array_equal(labels[1], expected[0])
    np.testing.assert_array_equal(labels[3], expected[1])
    assert 0 not in stationary_event_calls


def test_empty_user_with_verbose(capsys, stationary_event_calls):
    labels = Infostop(verbose=True).fit_predict([TRACE_1, np.empty((0, 3))])
    assert len(labels[1]) == 0
    assert 0 not in stationary_event_calls


@pytest.mark.parametrize("data", [[np.empty((0, 3))] * 2, np.empty((0, 3))])
def test_only_empty_traces(data, stationary_event_calls):
    with pytest.raises(Exception, match="^No stop events found"):
        Infostop().fit_predict(data)
    assert stationary_event_calls == []
