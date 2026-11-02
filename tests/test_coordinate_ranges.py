"""The coordinate checks of `Infostop` and `SpatialInfomap` with 'haversine' (#102).

Latitudes from -90 to 90 and longitudes from -180 to 180 are valid, both ends
included. infostop itself returns 180.0 for a place on the date line (#81, #96), and
GPS data rounded to 5 decimals has +-180.0 for points within half a metre of it.
"""

import numpy as np
import pytest

from infostop import Infostop
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
    assert np.array_equal(model.compute_label_medians()[labels[0]], [lat, lon])


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


def test_minus_180_and_180_are_one_place():
    # Visits to a place on the line, with points at 179.99999 and at the line itself,
    # and to a place 100 m away, alternating. The line is spelled -180 or 180.
    rng = np.random.default_rng(0)
    line = rng.choice([179.99999, -180.0], size=20)
    away = np.full(20, 179.999)
    lons = np.concatenate([line, away, line, away])
    trace_minus = np.column_stack(
        [np.full(len(lons), -16.8), lons, 60.0 * np.arange(len(lons))]
    )
    trace_plus = trace_minus.copy()
    trace_plus[trace_plus[:, 1] == -180.0, 1] = 180.0

    model_minus, model_plus = Infostop(), Infostop()
    labels_minus = model_minus.fit_predict(trace_minus)
    labels_plus = model_plus.fit_predict(trace_plus)

    assert labels_minus[0] >= 0
    assert np.all(labels_minus[:20] == labels_minus[0])
    assert np.array_equal(labels_minus, labels_plus)
    medians_minus = model_minus.compute_label_medians()
    medians_plus = model_plus.compute_label_medians()
    assert medians_minus.keys() == medians_plus.keys()
    for label, (lat, lon) in medians_minus.items():
        # The same longitude up to its spelling: -180 and 180 are one meridian
        assert lat == medians_plus[label][0]
        assert (lon - medians_plus[label][1]) % 360 == 0

    spatial_minus = SpatialInfomap().fit_predict(trace_minus[:, :2])
    spatial_plus = SpatialInfomap().fit_predict(trace_plus[:, :2])
    assert np.array_equal(spatial_minus, spatial_plus)


def test_a_stay_at_the_north_pole_with_any_longitudes_is_one_stop():
    # At the pole, every longitude is the same point
    lons = np.tile([-180.0, -90.0, 0.0, 90.0, 180.0], 2)
    trace = np.column_stack([np.full(10, 90.0), lons, 60.0 * np.arange(10)])

    labels = Infostop().fit_predict(trace)

    assert labels[0] >= 0
    assert np.all(labels == labels[0])
