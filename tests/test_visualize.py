import re

import folium
import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap, plot_map

DISPLAY_DATA = ["all", "all_stationary", "all_nonstationary", "unique_stationary"]

HOME = (55.70, 12.50)
WORK = (55.71, 12.52)
SHOP = (55.73, 12.56)


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


def _trace(rng, first=HOME, second=WORK):
    """First, second, first: 90 stationary points in 3 stop events, 20 moving points.

    The way back runs ~220 m north of the way out, so no two moving points are close.
    """
    north = (2e-3, 0)
    return np.vstack(
        [
            _visit(rng, first, 0),
            _move(rng, first, second, 2000),
            _visit(rng, second, 4000),
            _move(rng, np.add(second, north), np.add(first, north), 6000),
            _visit(rng, first, 8000),
        ]
    )


def _fitted(kind):
    """Return a fitted model, its input as a list of traces, and their labels."""
    rng = np.random.default_rng(0)
    if kind == "infostop":
        model, data = Infostop(), [_trace(rng)]
    elif kind == "infostop_no_timestamps":
        model, data = Infostop(), [_trace(rng)[:, :2]]
    elif kind == "infostop_multiuser":
        model, data = Infostop(), [_trace(rng), _trace(rng, WORK, HOME)]
    elif kind == "spatial_infomap":
        # The moving points are singletons, which `label_singleton=False` labels -1
        model, data = SpatialInfomap(label_singleton=False), [_trace(rng)[:, :2]]
    elif kind == "spatial_infomap_rounded":
        # Fitted on rounded coordinates, so `_stat_coords` differ from the input
        model = SpatialInfomap(
            label_singleton=False, min_spacial_resolution=2e-5, r2=30
        )
        data = [_trace(rng)[:, :2]]
    if kind.startswith("infostop"):
        labels = model.fit_predict(data if len(data) > 1 else data[0])
        labels = labels if len(data) > 1 else [labels]
    else:
        labels = [model.fit_predict(data[0])]
    return model, data, labels


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
    "spatial_infomap_rounded": {
        "all": 110,
        "all_stationary": 90,
        "all_nonstationary": 20,
        "unique_stationary": 27,
    },
}


@pytest.mark.parametrize("display_data", DISPLAY_DATA)
@pytest.mark.parametrize("kind", list(EXPECTED_COUNTS))
def test_plot_map_returns_folium_map(kind, display_data):
    model, _, _ = _fitted(kind)
    folmap = plot_map(model, display_data=display_data, scatter=True)
    assert isinstance(folmap.m, folium.Map)
    assert "<html>" in folmap.m.get_root().render()


@pytest.mark.parametrize("kind", list(EXPECTED_COUNTS))
def test_plot_map_display_data_selects_points(kind):
    model, _, _ = _fitted(kind)
    for display_data, count in EXPECTED_COUNTS[kind].items():
        folmap = plot_map(model, display_data=display_data)
        assert len(folmap.points) == len(folmap.labels) == count, display_data
        assert folmap.points.shape[1] == 2, display_data
        if display_data == "all_nonstationary":
            assert np.all(folmap.labels == -1)
        elif display_data != "all":
            assert np.all(folmap.labels >= 0), display_data


@pytest.mark.parametrize("kind", list(EXPECTED_COUNTS))
def test_plot_map_display_data_points_match_input_labels(kind):
    model, data, labels = _fitted(kind)
    expected = {
        (lat, lon, label)
        for data_u, labels_u in zip(data, labels)
        for (lat, lon), label in zip(data_u[:, :2].tolist(), labels_u.tolist())
    }
    for display_data, keep in [
        ("all", lambda label: True),
        ("all_stationary", lambda label: label >= 0),
        ("all_nonstationary", lambda label: label == -1),
    ]:
        folmap = plot_map(model, display_data=display_data)
        shown = list(zip(*folmap.points.T.tolist(), folmap.labels.tolist()))
        assert len(shown) == len(set(shown)), display_data
        assert set(shown) == {p for p in expected if keep(p[2])}, display_data


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

    for display_data, expected_labels in [
        ("all", [-1, labels[0]]),
        ("all_stationary", [labels[0]]),
        ("all_nonstationary", [-1]),
    ]:
        folmap = plot_map(model, display_data=display_data)
        at_point = np.all(folmap.points == trace[0, :2], axis=1)
        assert sorted(folmap.labels[at_point]) == expected_labels, display_data


def _nothing_to_show(case):
    """Return a fitted model and the `display_data` options that select no points."""
    # Three places visited once, with a long stay at home: the median and the mean of
    # the stop events, and the median of all points, are three different places
    rng = np.random.default_rng(0)
    visited_once = np.vstack(
        [
            _visit(rng, HOME, 0, n=60),
            _visit(rng, WORK, 4000, n=10),
            _visit(rng, SHOP, 6000, n=10),
        ]
    )
    if case == "infostop_all_-1":
        model = Infostop(label_singleton=False)
        model.fit_predict(visited_once)
        return model, {"all_stationary", "unique_stationary"}
    elif case == "infostop_no_moving_points":
        model = Infostop()
        model.fit_predict(visited_once)
        return model, {"all_nonstationary"}
    elif case == "spatial_infomap_all_-1":
        model = SpatialInfomap(label_singleton=False)
        model.fit_predict(np.array([HOME, WORK, SHOP]))
        return model, {"all_stationary", "unique_stationary"}
    elif case == "spatial_infomap_no_-1":
        model = SpatialInfomap()
        model.fit_predict(visited_once[:, :2])
        return model, {"all_nonstationary"}


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "case",
    [
        "infostop_all_-1",
        "infostop_no_moving_points",
        "spatial_infomap_all_-1",
        "spatial_infomap_no_-1",
    ],
)
def test_plot_map_with_nothing_to_show(case):
    model, empty = _nothing_to_show(case)
    for display_data in DISPLAY_DATA:
        folmap = plot_map(
            model,
            display_data=display_data,
            scatter=True,
            scatter_subsampling=0.5,
            heatmap_subsampling=0.5,
        )
        assert isinstance(folmap.m, folium.Map)
        assert "<html>" in folmap.m.get_root().render()
        if display_data in empty:
            # Centered on all the coordinates the model clustered
            assert len(folmap.points) == 0, display_data
            center = np.median(model._stat_coords, axis=0)
        else:
            assert len(folmap.points) > 0, display_data
            center = np.median(folmap.points, axis=0)
        assert np.allclose(folmap.m.location, center, rtol=0, atol=1e-9), display_data


def test_plot_map_keeps_the_tile_attribution():
    model, _, _ = _fitted("infostop")
    html = plot_map(model).m.get_root().render()
    assert "openstreetmap.org/copyright" in html
    assert "Infostop" not in html


def test_plot_map_with_custom_tile_url():
    # folium requires an attribution for a custom tile URL
    model, _, _ = _fitted("infostop")
    tiles = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    folmap = plot_map(model, tiles=tiles, zoom_start=15)
    html = folmap.m.get_root().render()
    assert tiles in html
    assert re.search(r"\bzoom\"?: 15\b", html)
    assert np.allclose(folmap.m.location, np.median(folmap.points, axis=0))


def test_plot_map_with_unknown_tiles():
    model, _, _ = _fitted("infostop")
    with pytest.raises(ValueError):
        plot_map(model, tiles="OpenStreetMapp")
