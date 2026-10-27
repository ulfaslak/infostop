import folium
import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap, plot_map

DISPLAY_DATA = ["all", "all_stationary", "all_nonstationary", "unique_stationary"]

HOME = (55.70, 12.50)
WORK = (55.71, 12.52)


def _visit(rng, location, t0, n=30):
    """`n` points scattered ~2 m around `location`, one per minute from `t0`."""
    return np.column_stack(
        [
            location[0] + rng.normal(0, 2e-5, n),
            location[1] + rng.normal(0, 2e-5, n),
            t0 + 60 * np.arange(n),
        ]
    )


def _move(rng, a, b, t0, n=10):
    """`n` points ~150 m apart on the line from `a` to `b`, one per minute from `t0`."""
    f = np.linspace(0, 1, n + 2)[1:-1]
    return np.column_stack(
        [
            a[0] + f * (b[0] - a[0]) + rng.normal(0, 2e-5, n),
            a[1] + f * (b[1] - a[1]) + rng.normal(0, 2e-5, n),
            t0 + 60 * np.arange(n),
        ]
    )


def _trace(rng):
    """Home, work, home: 90 stationary points in 3 stop events, 20 moving points.

    The way back runs ~220 m north of the way out, so no two moving points are close.
    """
    north = (2e-3, 0)
    return np.vstack(
        [
            _visit(rng, HOME, 0),
            _move(rng, HOME, WORK, 2000),
            _visit(rng, WORK, 4000),
            _move(rng, np.add(WORK, north), np.add(HOME, north), 6000),
            _visit(rng, HOME, 8000),
        ]
    )


def _fitted(kind):
    rng = np.random.default_rng(0)
    if kind == "infostop":
        model = Infostop()
        model.fit_predict(_trace(rng))
    elif kind == "infostop_no_timestamps":
        model = Infostop()
        model.fit_predict(_trace(rng)[:, :2])
    elif kind == "infostop_multiuser":
        model = Infostop()
        model.fit_predict([_trace(rng), _trace(rng)])
    elif kind == "spatial_infomap":
        # The moving points are singletons, which `label_singleton=False` labels -1
        model = SpatialInfomap(label_singleton=False)
        model.fit_predict(_trace(rng)[:, :2])
    return model


# Number of points shown for each `display_data`
EXPECTED_COUNTS = {
    "infostop": {
        "all": 110,
        "all_stationary": 90,
        "all_nonstationary": 20,
        "unique_stationary": 3,
    },
    "infostop_no_timestamps": {
        "all": 110,
        "all_stationary": 90,
        "all_nonstationary": 20,
        "unique_stationary": 3,
    },
    "infostop_multiuser": {
        "all": 220,
        "all_stationary": 180,
        "all_nonstationary": 40,
        "unique_stationary": 6,
    },
    "spatial_infomap": {
        "all": 110,
        "all_stationary": 90,
        "all_nonstationary": 20,
        "unique_stationary": 90,
    },
}


@pytest.mark.parametrize("display_data", DISPLAY_DATA)
@pytest.mark.parametrize("kind", list(EXPECTED_COUNTS))
def test_plot_map_returns_folium_map(kind, display_data):
    folmap = plot_map(_fitted(kind), display_data=display_data, scatter=True)
    assert isinstance(folmap.m, folium.Map)
    assert "<html>" in folmap.m.get_root().render()


@pytest.mark.parametrize("kind", list(EXPECTED_COUNTS))
def test_plot_map_display_data_selects_points(kind):
    model = _fitted(kind)
    for display_data, count in EXPECTED_COUNTS[kind].items():
        folmap = plot_map(model, display_data=display_data)
        assert len(folmap.points) == len(folmap.labels) == count, display_data
        assert folmap.points.shape[1] == 2, display_data
        if display_data == "all_nonstationary":
            assert np.all(folmap.labels == -1)
        elif display_data != "all":
            assert np.all(folmap.labels >= 0), display_data


def test_plot_map_display_data_points_match_input_labels():
    rng = np.random.default_rng(0)
    trace = _trace(rng)
    model = Infostop()
    labels = model.fit_predict(trace)
    shown = {}
    for display_data in ["all", "all_stationary", "all_nonstationary"]:
        folmap = plot_map(model, display_data=display_data)
        shown[display_data] = {
            (lat, lon, label)
            for (lat, lon), label in zip(folmap.points.tolist(), folmap.labels.tolist())
        }
    expected = {
        (lat, lon, label) for (lat, lon), label in zip(trace[:, :2].tolist(), labels)
    }
    assert shown["all"] == expected
    assert shown["all_stationary"] == {p for p in expected if p[2] >= 0}
    assert shown["all_nonstationary"] == {p for p in expected if p[2] == -1}


def test_plot_map_shows_a_point_once_for_every_label_it_has():
    # A moving point that lands exactly on a point of the first home visit
    rng = np.random.default_rng(0)
    trace = _trace(rng)
    away = _move(rng, HOME, WORK, trace[-1, 2] + 60, n=1)
    revisit = np.array([[trace[0, 0], trace[0, 1], trace[-1, 2] + 120]])
    trace = np.vstack([trace, away, revisit, away + [0, 0, 120]])
    model = Infostop()
    labels = model.fit_predict(trace)
    assert labels[0] >= 0 and labels[-2] == -1

    for display_data in ["all", "all_stationary", "all_nonstationary"]:
        points = plot_map(model, display_data=display_data).points
        assert np.any(np.all(points == trace[0, :2], axis=1)), display_data
