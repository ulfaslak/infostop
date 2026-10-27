import numpy as np
import pytest

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


# Two users in different neighbourhoods (about 10 km apart), so they share no stops.
# The last visit of WITH_TIME lasts 120 s, under the default `min_staying_time`: it
# is a stop only if the time column is ignored.
A, B, C = (55.70, 12.50), (55.71, 12.52), (55.70, 12.53)
D, E = (55.80, 12.40), (55.81, 12.42)
WITH_TIME = make_trace(0, [A, B, A, C, A], [30, 20, 25, 15, 3])
WITHOUT_TIME = make_trace(1, [D, E, D, E, D], [20, 30, 10, 25, 15])[:, :2]


def same_partition(a, b):
    """Equal up to relabelling, with -1 in the same positions."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a == -1, b == -1):
        return False
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("label_singleton", [True, False])
@pytest.mark.parametrize("distance_metric", ["haversine", "euclidean"])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_mixed_column_users_match_separate_fits(
    order, distance_metric, label_singleton
):
    traces = [WITH_TIME, WITHOUT_TIME]
    if distance_metric == "euclidean":  # degrees to metres-ish, keep time as is
        traces = [t * [1e5, 1e5, 1][: t.shape[1]] for t in traces]
    traces = [traces[i] for i in order]
    kw = dict(distance_metric=distance_metric, label_singleton=label_singleton)

    labels = Infostop(**kw).fit_predict(traces)

    assert [len(x) for x in labels] == [len(t) for t in traces]
    for labels_u, trace in zip(labels, traces):
        alone = Infostop(**kw).fit_predict(trace)
        assert same_partition(labels_u, alone)
        assert -1 in labels_u and len(set(labels_u) - {-1}) >= 1
        if trace.shape[1] == 3:  # the time column changes this user's labels
            assert not same_partition(alone, Infostop(**kw).fit_predict(trace[:, :2]))
    # No stop is shared between the two neighbourhoods
    assert not (set(labels[0]) & set(labels[1])) - {-1}


def test_stop_shared_by_user_with_and_user_without_time():
    # Rows: A 0-29 | moving | B 40-59 | moving | A 70-94
    with_time = make_trace(2, [A, B, A], [30, 20, 25])
    # Rows: A 0-19 | moving | C 30-59 | moving | A 70-84
    without_time = make_trace(3, [A, C, A], [20, 30, 15])[:, :2]

    labels_t, labels_u = Infostop().fit_predict([with_time, without_time])

    def stops(rows):  # a visit's edge point can be -1 when noise exceeds `r1`
        return set(rows) - {-1}

    at_a = stops(labels_t[:30]) | stops(labels_t[70:])
    at_a |= stops(labels_u[:20]) | stops(labels_u[70:])
    assert len(at_a) == 1  # one shared stop at A
    (b,), (c,) = stops(labels_t[40:60]), stops(labels_u[30:60])
    assert len({*at_a, b, c}) == 3  # B and C are separate stops
    assert b not in labels_u and c not in labels_t


def test_mixed_column_users_nan_count():
    with_time, without_time = WITH_TIME.copy(), WITHOUT_TIME.copy()
    with_time[3, 2] = np.nan
    without_time[4, :] = np.nan
    with pytest.raises(
        AssertionError, match=r"^There are 3 NaN values in the input data\.$"
    ):
        Infostop().fit_predict([with_time, without_time])


def test_mixed_column_users_column_check_still_applies():
    four_columns = np.column_stack([WITH_TIME, WITH_TIME[:, 2]])
    with pytest.raises(
        AssertionError, match=r"^User 1: Number of columns must be 2 or 3$"
    ):
        Infostop().fit_predict([WITHOUT_TIME, four_columns])


def test_no_traces():
    with pytest.raises(
        AssertionError, match=r"^There are no traces in the input data\.$"
    ):
        Infostop().fit_predict([])
