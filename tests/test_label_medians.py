"""`Infostop.compute_label_medians` and its helper `utils.median_lat_lon` (#96).

A label median is the coordinate-wise median of the label's stop event medians. With
'haversine', a label whose longitudes span more than 180 degrees, and less once the
negative ones are moved up by 360, is taken to lie on both sides of longitude +-180,
and its median longitude is taken across that line.
"""

import numpy as np
import pytest

from infostop import Infostop, utils


def taveuni(first_lon, second_lon, gap=0):
    """Two visits to one place on Taveuni (Fiji), which the 180th meridian crosses.

    Each visit is 10 minutes at the place and a 3-point walk away from it. With
    `gap`, the walk is replaced by `gap` seconds without points.
    """
    rows, t = [], 0
    for lon in (first_lon, second_lon):
        for _ in range(10):
            rows.append([-16.8, lon, t])
            t += 60
        if gap:
            t += gap
            continue
        for i in range(3):
            rows.append([-16.8 + 0.01 * (i + 1), lon, t])
            t += 60
    return np.array(rows, dtype=float)


@pytest.mark.parametrize(
    "first_lon, second_lon",
    [(179.99999, -179.99999), (-179.99999, 179.99999)],
    ids=["east-first", "west-first"],
)
def test_label_median_of_a_place_on_the_antimeridian(first_lon, second_lon):
    model = Infostop()
    labels = model.fit_predict(taveuni(first_lon, second_lon))
    assert len(np.unique(labels[labels >= 0])) == 1

    ((lat, lon),) = model.compute_label_medians().values()

    assert lat == -16.8
    assert abs(abs(lon) - 180) < 1e-4


def _stays(rng, lon0, n_places=4, n_visits=10):
    """Visits to a few places around (40.7, lon0), 5-decimal coordinates, with walks."""
    places = np.column_stack(
        [40.7 + rng.normal(0, 0.002, n_places), lon0 + rng.normal(0, 0.002, n_places)]
    )
    rows, t = [], 0.0
    for p in rng.integers(0, n_places, n_visits):
        for _ in range(int(rng.integers(6, 15))):
            rows.append([*np.round(places[p] + rng.normal(0, 2e-5, 2), 5), t])
            t += 60
        for _ in range(3):
            rows.append([*np.round(places[p] + rng.normal(0, 0.01, 2), 5), t])
            t += 60
    return np.array(rows)


@pytest.mark.parametrize("lon0", [-73.98, -0.0005, 12.57, 179.0, -179.0])
def test_label_medians_away_from_the_antimeridian_are_the_plain_median(lon0):
    # Bit for bit: shifting western longitudes by 360 and back would change their
    # last bits.
    rng = np.random.default_rng(96)
    model = Infostop(min_spacial_resolution=1e-5)
    model.fit_predict(_stays(rng, lon0))
    stat_coords, stat_labels = model._stat_coords, model._stat_labels

    medians = model.compute_label_medians()

    assert len(medians) > 1
    for label, median in medians.items():
        plain = np.median(stat_coords[stat_labels == label], axis=0).tolist()
        assert median == plain, label


def test_euclidean_label_medians_are_not_wrapped():
    # With 'euclidean', column 1 is not a longitude: a label that spans more than
    # 180 units keeps its plain median.
    rows, t = [], 0
    for y in (-200.0, 200.0, -200.0):
        for _ in range(10):
            rows.append([0.0, y, t])
            t += 60
        rows.append([1000.0, 1000.0 + y, t])
        t += 60
    model = Infostop(distance_metric="euclidean", r2=400)
    labels = model.fit_predict(np.array(rows))
    assert len(np.unique(labels[labels >= 0])) == 1

    # The median of the label's two distinct event coordinates, (0, -200) and (0, 200)
    assert list(model.compute_label_medians().values()) == [[0.0, 0.0]]


@pytest.mark.parametrize(
    "lons, expected",
    [
        ([179.99999, -179.99999], 180.0),
        ([179.9, -179.7, -179.8], -179.8),
        ([-179.9, 179.8, 179.7], 179.8),
        ([180.0, -180.0], 180.0),
        ([-180.0, -180.0], -180.0),
        # San Francisco and Tokyo: closer across the Pacific than across Greenwich
        ([139.69, -122.42], -171.365),
    ],
)
def test_median_lat_lon_across_the_antimeridian(lons, expected):
    coords = np.column_stack([np.full(len(lons), 10.0), lons])

    lat, lon = utils.median_lat_lon(coords)

    assert lat == 10.0
    assert lon == pytest.approx(expected, abs=1e-9)
    assert -180 <= lon <= 180


@pytest.mark.parametrize(
    "lons",
    [
        [-73.98121, -73.98122],
        [-73.98121, -73.98123, -73.98122],
        [-0.00057, 0.00001, -0.00003, 0.00002],
        [12.5, 12.6, 12.55],
        [-90.0, 90.0],  # spans exactly 180: not shifted
    ],
)
def test_median_lat_lon_is_np_median_up_to_a_span_of_180(lons):
    coords = np.column_stack([np.linspace(-80, 80, len(lons)), lons])

    assert np.array_equal(utils.median_lat_lon(coords), np.median(coords, axis=0))


@pytest.mark.parametrize(
    "lons",
    [
        [139.69, -74.01, -0.13],  # Tokyo, New York, London
        [-170.0, -10.0, 10.0, 170.0],
    ],
)
def test_median_lat_lon_of_points_around_the_globe_is_np_median(lons):
    # The longitudes span more than 180, but not because they lie on both sides of
    # +-180: moving the negative ones up by 360 does not make the span smaller.
    coords = np.column_stack([np.linspace(-80, 80, len(lons)), lons])

    assert np.array_equal(utils.median_lat_lon(coords), np.median(coords, axis=0))


def test_median_lat_lon_of_float32_across_the_antimeridian_is_taken_in_float64():
    # A float32 longitude plus 360 would lose its last digits
    coords = np.array(
        [[-16.8, 179.99999], [-16.8, -179.99998], [-16.8, -179.99999]], dtype=np.float32
    )

    lat, lon = utils.median_lat_lon(coords)

    assert lon == utils.median_lat_lon(coords.astype(float))[1]
