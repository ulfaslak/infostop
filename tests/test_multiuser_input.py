import numpy as np
import pytest

from infostop import Infostop


def make_trace(seed=0):
    """Home, work, home: two Gaussian blobs of points, 60 s apart within a visit."""
    rng = np.random.default_rng(seed)

    def visit(lat, lon, t0, n=30):
        return np.column_stack(
            [
                lat + rng.normal(0, 2e-5, n),
                lon + rng.normal(0, 2e-5, n),
                t0 + 60 * np.arange(n),
            ]
        )

    return np.vstack(
        [visit(55.70, 12.50, 0), visit(55.71, 12.52, 4000), visit(55.70, 12.50, 8000)]
    )


@pytest.mark.parametrize("n_users", [1, 2, 3])
def test_tuple_gives_same_labels_as_list(n_users):
    traces = [make_trace(seed) for seed in range(n_users)]
    labels_list = Infostop().fit_predict(list(traces))
    labels_tuple = Infostop().fit_predict(tuple(traces))
    assert type(labels_tuple) is tuple
    assert len(labels_tuple) == n_users
    for a, b, trace in zip(labels_list, labels_tuple, traces):
        assert isinstance(b, np.ndarray)
        assert len(b) == len(trace)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("container", [list, tuple])
def test_return_type_matches_input_type(container):
    traces = container([make_trace(0), make_trace(1)])
    assert type(Infostop().fit_predict(traces)) == type(traces)
    trace = make_trace(0)
    assert type(Infostop().fit_predict(trace)) == type(trace)


def test_list_input_returns_model_labels():
    model = Infostop()
    labels = model.fit_predict([make_trace(0), make_trace(1)])
    assert labels is model.labels


@pytest.mark.parametrize(
    "data, got",
    [
        # A single trace as a plain list of [lat, lon, t] rows is read as one user
        # per row. It must fail clearly, not be reinterpreted as one trace.
        (make_trace().tolist(), r"User 0: .*got a list\."),
        (tuple(map(tuple, make_trace())), r"User 0: .*got a tuple\."),
        # Several users, each a list of rows
        ([make_trace(0).tolist(), make_trace(1).tolist()], r"User 0: .*got a list\."),
        # Columns passed separately
        (tuple(make_trace().T), r"User 0: .*got an array of shape \(90,\)\."),
        # A single row
        (make_trace()[0], r"^Each .*got an array of shape \(3,\)\."),
        # Users stacked into one 3D array
        (np.stack([make_trace(0), make_trace(1)]), r"shape \(2, 90, 3\)\."),
    ],
)
def test_input_that_is_not_a_2d_array_fails_clearly(data, got):
    with pytest.raises(AssertionError, match=got) as excinfo:
        Infostop().fit_predict(data)
    assert "must be a 2D numpy array with one row per point" in str(excinfo.value)
