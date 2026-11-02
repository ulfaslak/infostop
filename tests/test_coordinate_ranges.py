"""The coordinate checks of `Infostop` and `SpatialInfomap` with 'haversine' (#102).

Latitudes from -90 to 90 and longitudes from -180 to 180 are valid, both ends
included. infostop itself returns 180.0 for a place on the date line (#81, #96), and
GPS data rounded to 5 decimals has +-180.0 for points within about half a metre of it.
-180 and 180 are one meridian, and every longitude at a pole is one point, so how a
place is written must not change the labels.
"""

import numpy as np
import pytest

from infostop import Infostop, utils
from infostop.models import SpatialInfomap

LAT_MESSAGE = r"Latitude \(column 0\) must have values between -90 and 90"
LON_MESSAGE = r"Longitude \(column 1\) must have values between -180 and 180"

ENDS = [(90.0, 0.0), (-90.0, 0.0), (-16.8, 180.0), (-16.8, -180.0)]
ENDS_IDS = ["lat=90", "lat=-90", "lon=180", "lon=-180"]


def stay_then_walk(lat, lon):
    """10 minutes at one coordinate, then a 3-point walk towards the equator."""
    step = -0.01 if lat > 0 else 0.01
    rows = [[lat, lon, 60 * i] for i in range(10)]
    rows += [[lat + step * (i + 1), lon, 600 + 60 * i] for i in range(3)]
    return np.array(rows, dtype=float)


def taveuni(first_lon, second_lon):
    """Two visits to one place on Taveuni (Fiji), each followed by a walk away."""
    rows, t = [], 0
    for lon in (first_lon, second_lon):
        for _ in range(10):
            rows.append([-16.8, lon, t])
            t += 60
        for i in range(3):
            rows.append([-16.8 + 0.01 * (i + 1), lon, t])
            t += 60
    return np.array(rows, dtype=float)


@pytest.mark.parametrize("lat, lon", ENDS, ids=ENDS_IDS)
def test_infostop_accepts_the_ends_of_the_ranges(lat, lon):
    model = Infostop()
    labels = model.fit_predict(stay_then_walk(lat, lon))

    assert labels[0] >= 0
    assert np.all(labels[:10] == labels[0])
    assert np.all(labels[10:] == -1)
    # infostop writes the place one way: longitude 180 for -180, and 0 at the poles
    expected = [lat, 0.0] if abs(lat) == 90 else [lat, 180.0 if lon == -180 else lon]
    assert np.array_equal(model.compute_label_medians()[labels[0]], expected)


@pytest.mark.parametrize("lat, lon", ENDS, ids=ENDS_IDS)
def test_spatial_infomap_accepts_the_ends_of_the_ranges(lat, lon):
    labels = SpatialInfomap().fit_predict(np.array([[lat, lon], [lat, lon]]))

    assert labels[0] >= 0
    assert labels[1] == labels[0]


OUTSIDE = [
    (0, np.nextafter(90.0, np.inf), LAT_MESSAGE),
    (0, np.nextafter(-90.0, -np.inf), LAT_MESSAGE),
    (1, np.nextafter(180.0, np.inf), LON_MESSAGE),
    (1, np.nextafter(-180.0, -np.inf), LON_MESSAGE),
]
OUTSIDE_IDS = ["lat>90", "lat<-90", "lon>180", "lon<-180"]


@pytest.mark.parametrize("column, value, message", OUTSIDE, ids=OUTSIDE_IDS)
def test_infostop_rejects_values_just_outside_the_ranges(column, value, message):
    trace = stay_then_walk(-16.8, 179.9)
    trace[4, column] = value
    with pytest.raises(AssertionError, match=f"^{message}$"):
        Infostop().fit_predict(trace)


@pytest.mark.parametrize("column, value, message", OUTSIDE, ids=OUTSIDE_IDS)
def test_infostop_multi_user_names_the_user_outside_the_ranges(column, value, message):
    trace = stay_then_walk(-16.8, 179.9)
    trace[4, column] = value
    with pytest.raises(AssertionError, match=f"^User 1: {message}$"):
        Infostop().fit_predict([stay_then_walk(-16.8, 179.9), trace])


@pytest.mark.parametrize("column, value, message", OUTSIDE, ids=OUTSIDE_IDS)
def test_spatial_infomap_rejects_values_just_outside_the_ranges(column, value, message):
    coords = stay_then_walk(-16.8, 179.9)[:, :2]
    coords[4, column] = value
    with pytest.raises(AssertionError, match=f"^{message}$"):
        SpatialInfomap().fit_predict(coords)


def test_gps_data_rounded_to_5_decimals_on_the_date_line():
    # A point 0.4 m east of the line becomes -180.0
    trace = np.array(
        [[-16.8, 179.99999, 0], [-16.8, -179.999996, 60], [-16.8, 179.99999, 120]]
    )
    trace[:, :2] = np.round(trace[:, :2], 5)
    assert trace[1, 1] == -180.0

    labels = Infostop(min_staying_time=60).fit_predict(trace)

    assert labels[0] >= 0
    assert np.all(labels == labels[0])


def test_label_medians_on_the_date_line_can_be_clustered_again():
    model = Infostop()
    model.fit_predict(taveuni(179.99999, -179.99999))
    medians = np.array(list(model.compute_label_medians().values()))
    assert medians.tolist() == [[-16.8, 180.0]]

    labels = SpatialInfomap().fit_predict(medians)

    assert labels.shape == (1,)
    assert labels[0] >= 0


def spell_line(coords, spelling):
    """`coords` with the date line written as 180, as -180, or each way in turn."""
    coords = np.array(coords, dtype=float)
    on_line = np.abs(coords[:, 1]) == 180
    if spelling == "both":
        coords[on_line, 1] = np.where(np.arange(on_line.sum()) % 2, -180.0, 180.0)
    else:
        coords[on_line, 1] = {"plus": 180.0, "minus": -180.0}[spelling]
    return coords


def grid_around_the_line(seed):
    """8 to 30 points on a 1e-5 degree grid (GPS data rounded to 5 decimals), within
    about 3 m of longitude 180 and 15 m of a latitude on Taveuni."""
    rng = np.random.default_rng(seed)
    n = rng.integers(8, 30)
    dlon = rng.integers(-3, 4, size=n) * 1e-5
    lat = -16.8 + rng.integers(-15, 16, size=n) * 1e-5
    lon = np.round(180.0 + dlon, 5)
    lon = np.where(lon > 180, lon - 360, lon)
    return np.column_stack([lat, lon])


def visits_around_the_line(seed):
    """Visits to 2 to 5 places within a few metres of longitude 180, in GPS data
    rounded to 5 decimals."""
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    centres = [
        (-16.8 + rng.integers(-3, 4) * 1e-5, 180.0 + rng.integers(-3, 4) * 1e-5)
        for _ in range(rng.integers(2, 6))
    ]
    for _ in range(rng.integers(3, 10)):
        clat, clon = centres[rng.integers(len(centres))]
        for _ in range(rng.integers(6, 15)):
            rows.append([clat + rng.normal(0, 1e-5), clon + rng.normal(0, 1e-5), t])
            t += 60
        t += 600
    trace = np.array(rows)
    trace[:, :2] = np.round(trace[:, :2], 5)
    trace[:, 1] = np.where(trace[:, 1] > 180, trace[:, 1] - 360, trace[:, 1])
    return trace


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("seed", range(3))
def test_spatial_infomap_labels_do_not_depend_on_how_the_date_line_is_written(
    seed, weighted
):
    coords = grid_around_the_line(seed)
    assert (np.abs(coords[:, 1]) == 180).sum() >= 2

    labels = {
        spelling: SpatialInfomap(r2=2, weighted=weighted).fit_predict(
            spell_line(coords, spelling)
        )
        for spelling in ("plus", "minus", "both")
    }

    assert np.array_equal(labels["minus"], labels["plus"])
    assert np.array_equal(labels["both"], labels["plus"])


def test_spatial_infomap_label_numbers_do_not_depend_on_how_the_date_line_is_written():
    plus = SpatialInfomap(r2=1).fit_predict(
        np.array([[-16.8, 180.0], [-16.8, 179.99995]])
    )
    minus = SpatialInfomap(r2=1).fit_predict(
        np.array([[-16.8, -180.0], [-16.8, 179.99995]])
    )
    assert np.array_equal(minus, plus)


def test_infostop_label_numbers_do_not_depend_on_how_the_date_line_is_written():
    # A stay on the line, then a stay 100 m west of it
    def trace(line):
        lons = [line] * 10 + [179.999] * 10
        return np.column_stack(
            [
                np.full(20, -16.8),
                lons,
                60.0 * np.arange(20) + 3000 * (np.arange(20) >= 10),
            ]
        )

    assert np.array_equal(
        Infostop().fit_predict(trace(-180.0)), Infostop().fit_predict(trace(180.0))
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("seed", [38, 0, 1])
def test_infostop_labels_do_not_depend_on_how_the_date_line_is_written(seed, weighted):
    trace = visits_around_the_line(seed)
    assert (np.abs(trace[:, 1]) == 180).sum() >= 2

    def fit(spelling):
        model = Infostop(r1=3, r2=3, min_staying_time=120, weighted=weighted)
        return model.fit_predict(spell_line(trace, spelling))

    plus = fit("plus")
    assert np.array_equal(fit("minus"), plus)
    assert np.array_equal(fit("both"), plus)


def test_a_trace_that_writes_the_date_line_as_minus_180_has_one_node_for_it():
    # The stop event of the first stay is -180.0. The second stay crosses the line, so
    # its stop event is written 180.0, as stop events across the line are.
    def stay(lons, t0):
        return [[-16.8, lon, t0 + 60 * i] for i, lon in enumerate(lons)]

    walk = [[-16.8 + 0.01 * (i + 1), 179.9, 600 + 60 * i] for i in range(3)]
    trace = np.array(
        stay([-180.0] * 10, 0) + walk + stay([179.99999, -180.0, -179.99999] * 4, 3600)
    )
    model = Infostop(weighted=True)
    labels = model.fit_predict(trace)

    assert model._stat_coords.tolist() == [[-16.8, 180.0]]
    assert np.array_equal(
        labels, Infostop(weighted=True).fit_predict(spell_line(trace, "plus"))
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_points_at_a_pole_are_one_node(weighted):
    model = SpatialInfomap(weighted=weighted)
    labels = model.fit_predict(np.array([[90.0, 0.0], [90.0, 90.0], [90.0, -180.0]]))

    assert model._stat_coords.tolist() == [[90.0, 0.0]]
    assert model._counts.tolist() == [3]
    assert labels[0] >= 0 and np.all(labels == labels[0])


@pytest.mark.parametrize(
    "lat, lon, median",
    [
        (-16.8, 179.999999, [-16.8, 180.0]),
        (-16.8, -179.999999, [-16.8, 180.0]),
        (89.999999, 10.0, [90.0, 0.0]),
        (-89.999999, 10.0, [-90.0, 0.0]),
    ],
)
def test_label_medians_with_min_spacial_resolution_can_be_clustered_again(
    lat, lon, median
):
    # Rounding by 1e-5 gives 180.00000000000003, -180.00000000000003 and
    # 90.00000000000001 here, which are past the ends of the ranges
    model = Infostop(min_spacial_resolution=1e-5)
    model.fit_predict(stay_then_walk(lat, lon))
    medians = np.array(list(model.compute_label_medians().values()))

    assert medians.tolist() == [median]
    assert SpatialInfomap().fit_predict(medians).tolist() == [0]


@pytest.mark.parametrize(
    "coords, expected",
    [
        ([[10.0, -180.0]], [[10.0, 180.0]]),
        ([[10.0, 180.0]], [[10.0, 180.0]]),
        ([[10.0, 180.00000000000003]], [[10.0, 180.0]]),
        ([[10.0, -180.00000000000003]], [[10.0, 180.0]]),
        ([[90.00000000000001, 10.0]], [[90.0, 0.0]]),
        ([[-90.0, -180.0]], [[-90.0, 0.0]]),
        ([[90.0, 0.0]], [[90.0, 0.0]]),
    ],
)
def test_canonical_lat_lon(coords, expected):
    assert utils.canonical_lat_lon(np.array(coords)).tolist() == expected


@pytest.mark.parametrize(
    "coords, expected",
    [
        # Past the ends by more than a rounding error: the same place inside them
        ([[0.0, 180.05]], [[0.0, -179.95]]),
        ([[0.0, -180.05]], [[0.0, 179.95]]),
        ([[90.3, 9.8]], [[89.7, -170.2]]),
        ([[-90.3, -9.8]], [[-89.7, 170.2]]),
        ([[90.3, -170.0]], [[89.7, 10.0]]),
    ],
)
def test_canonical_lat_lon_past_the_ends(coords, expected):
    assert np.allclose(utils.canonical_lat_lon(np.array(coords)), expected, atol=1e-9)


def _haversine(a, b):
    a, b = np.radians(a), np.radians(b)
    h = (
        np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
        + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    )
    return 2 * 6371000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def test_canonical_lat_lon_does_not_move_points_on_the_globe():
    rng = np.random.default_rng(0)
    # Points past the ends by up to 0.5 degrees, as rounding by 1 can give
    coords = np.column_stack(
        [rng.uniform(-90.5, 90.5, 200), rng.uniform(-180.5, 180.5, 200)]
    )
    others = np.column_stack([rng.uniform(-90, 90, 200), rng.uniform(-180, 180, 200)])
    canonical = utils.canonical_lat_lon(coords)

    assert np.all(np.abs(canonical[:, 0]) <= 90)
    assert np.all((canonical[:, 1] > -180) & (canonical[:, 1] <= 180))
    assert np.allclose(
        _haversine(canonical, others), _haversine(coords, others), atol=1e-6
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_canonical_lat_lon_keeps_the_dtype(dtype):
    coords = np.array([[10, -180], [90, 45], [20, 30]], dtype=dtype)
    canonical = utils.canonical_lat_lon(coords)
    assert canonical.dtype == dtype
    assert canonical.tolist() == [[10, 180], [90, 0], [20, 30]]


def test_canonical_lat_lon_returns_points_inside_the_ranges_as_they_are():
    rng = np.random.default_rng(0)
    coords = np.column_stack(
        [rng.uniform(-89.9, 89.9, 50), rng.uniform(-179.9, 179.9, 50)]
    )
    coords = np.vstack([coords, [[0.0, 180.0], [89.99, -179.99]]])
    for dtype in (np.float64, np.float32):
        c = coords.astype(dtype)
        assert utils.canonical_lat_lon(c) is c


def test_spatial_infomap_float32_labels_do_not_depend_on_how_the_date_line_is_written():
    plus = np.array([[-16.8, 180.0], [-16.8, 179.99995]], dtype=np.float32)
    minus = plus.copy()
    minus[0, 1] = -180.0
    assert np.array_equal(
        SpatialInfomap(r2=5).fit_predict(minus), SpatialInfomap(r2=5).fit_predict(plus)
    )
    for seed in range(20):
        coords = grid_around_the_line(seed).astype(np.float32)
        labels = [
            SpatialInfomap(r2=2).fit_predict(spell_line(coords, s).astype(np.float32))
            for s in ("plus", "minus", "both")
        ]
        assert np.array_equal(labels[1], labels[0]), seed
        assert np.array_equal(labels[2], labels[0]), seed


def test_rounding_past_the_date_line_keeps_places_apart():
    # Rounded by 0.65, 179.9 becomes 180.05 and -179.95 becomes -180.05: two places
    # 11 km apart, which must not become one at 180
    coords = np.array([[0.0, 179.9]] * 2 + [[0.0, -179.95]] * 2)
    labels = SpatialInfomap(r2=100, min_spacial_resolution=0.65).fit_predict(coords)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def test_rounding_past_a_pole_keeps_one_place_together():
    # Rounded by 0.7, (89.99, 10) becomes (90.3, 9.8): the same place as
    # (89.7, -170.2), which (89.75, -170.2) becomes
    coords = np.array([[89.99, 10.0]] * 2 + [[89.75, -170.2]] * 2)
    labels = SpatialInfomap(r2=20000, min_spacial_resolution=0.7).fit_predict(coords)
    assert np.all(labels == labels[0])


def test_a_label_median_at_a_pole_has_longitude_0():
    # A stay at the pole and a stay a last digit below it, with another longitude
    def stay(lat, lon, t0):
        return [[lat, lon, t0 + 60 * i] for i in range(10)]

    walk = [[89.99 - 0.01 * i, 0.0, 700 + 60 * i] for i in range(3)]
    trace = np.array(stay(90.0, 0.0, 0) + walk + stay(np.nextafter(90, 0), 45.0, 4000))
    model = Infostop()
    labels = model.fit_predict(trace)
    assert len(set(labels[labels >= 0])) == 1

    assert list(model.compute_label_medians().values()) == [[90.0, 0.0]]


def test_a_stay_at_the_north_pole_with_any_longitudes_is_one_stop():
    # At the pole, every longitude is the same point
    lons = np.tile([-180.0, -90.0, 0.0, 90.0, 180.0], 2)
    trace = np.column_stack([np.full(10, 90.0), lons, 60.0 * np.arange(10)])

    labels = Infostop().fit_predict(trace)

    assert labels[0] >= 0
    assert np.all(labels == labels[0])
