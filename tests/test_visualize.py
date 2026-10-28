import inspect
import json
import re

import folium
import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap, plot_map, visualize
from infostop.visualize import FoliumMap

DISPLAY_DATA = ["all", "all_stationary", "all_nonstationary", "unique_stationary"]

HOME = (55.70, 12.50)
WORK = (55.71, 12.52)
SHOP = (55.73, 12.56)

# folium 0.7 and 0.8 have their own Mapbox tiles, which take `API_key`
OLD_FOLIUM = "API_key" in inspect.signature(folium.Map).parameters


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


def _credits(html):
    """The tile credits (attributions) in a rendered page."""
    found = re.findall(r"\"?attribution\"?: *\"((?:[^\"\\]|\\.)*)\"", html)
    return [json.loads(f'"{credit}"') for credit in found]


def _render(**kwargs):
    model, _, _ = _fitted("infostop")
    return plot_map(model, **kwargs).m.get_root().render()


@pytest.mark.parametrize(
    "tiles, credit",
    [
        ("https://tile.opentopomap.org/{z}/{x}/{y}.png", "tile.opentopomap.org"),
        (
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "tile.openstreetmap.org",
        ),
        ("http://localhost:8080/{z}/{x}/{y}.png", "localhost:8080"),
        (
            "https://me:secret@tiles.example.com/{z}/{x}/{y}.png?k=secret",
            "tiles.example.com",
        ),
    ],
)
@pytest.mark.parametrize("attr", [None, "", "© OpenTopoMap (CC-BY-SA)"])
def test_plot_map_credit_of_a_tile_url(tiles, credit, attr):
    # A tile URL is credited with `attr`, or else with its host, not with infostop
    html = _render(tiles=tiles, attr=attr)
    assert _credits(html) == [attr or f"Tiles from {credit}"]
    assert "Infostop" not in html
    assert html.count("secret") == tiles.count("secret")


def test_plot_map_credit_of_a_tile_url_without_host():
    assert _credits(_render(tiles="/tiles/{z}/{x}/{y}.png")) == ["Custom tiles"]


@pytest.mark.filterwarnings("ignore:CartoDB tiles now require an API key")
@pytest.mark.parametrize("tiles", ["OpenStreetMap", "CartoDB positron"])
def test_plot_map_passes_attr_to_folium(tiles):
    # Named tiles get the credit that folium gives them with this `attr`: folium
    # 0.9 to 0.14 keep the tile set's own credit
    expected = folium.Map(location=HOME, tiles=tiles, attr="My credit").get_root()
    assert _credits(_render(tiles=tiles, attr="My credit")) == _credits(
        expected.render()
    )


def test_plot_map_attr_without_tiles():
    def page(**kwargs):
        # folium gives each element a random id
        return re.sub(r"_[0-9a-f]{32}", "_ID", _render(tiles=None, **kwargs))

    assert page(attr="My credit") == page()


@pytest.mark.skipif(OLD_FOLIUM, reason="folium < 0.9 does not use the key here")
def test_plot_map_attr_with_api_key():
    pytest.importorskip("xyzservices")
    html = _render(
        tiles="Thunderforest.OpenCycleMap", API_key="my-token", attr="My credit"
    )
    assert _credits(html) == ["My credit"]
    assert html.count("my-token") == 1


def _named_tiles(documented):
    """The quoted tile names in the `tiles` entry of a docstring."""
    lines = inspect.getdoc(documented).splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip().startswith("tiles :")
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    entry = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        entry.append(line)
    return re.findall(r"'([^']+)'", " ".join(entry))


@pytest.mark.parametrize("documented", [plot_map, FoliumMap])
def test_tiles_named_in_the_docs_need_no_key(documented):
    # The docs once recommended CartoDB tiles, which now show only "API KEY
    # REQUIRED" without a key
    xyzservices = pytest.importorskip("xyzservices")
    names = _named_tiles(documented)
    assert "OpenStreetMap" in names
    for name in names:
        # folium looks up "OpenStreetMap" under this name
        query = "OpenStreetMap Mapnik" if name == "OpenStreetMap" else name
        assert not xyzservices.providers.query_name(query).requires_token(), name


@pytest.mark.parametrize("attr", [None, "My credit"])
def test_plot_map_with_unknown_tiles(attr):
    model, _, _ = _fitted("infostop")
    with pytest.raises(ValueError):
        plot_map(model, tiles="OpenStreetMapp", attr=attr)


@pytest.mark.skipif(OLD_FOLIUM, reason="folium < 0.9 uses its own Mapbox tiles")
def test_plot_map_mapbox_api_key():
    xyzservices = pytest.importorskip("xyzservices")
    mapbox = xyzservices.providers.MapBox(accessToken="my-token")
    model, _, _ = _fitted("infostop")
    folmap = plot_map(model, tiles="Mapbox", API_key="my-token")
    # The layer control shows the name of the tile layer
    folium.LayerControl().add_to(folmap.m)
    html = folmap.m.get_root().render()
    tile_urls = re.findall(r"L\.tileLayer\(\s*[\"']([^\"']+)", html)
    assert tile_urls == [mapbox.build_url(fill_subdomain=False, scale_factor="{r}")]
    # The key is only in the tile URL
    assert html.count("my-token") == 1
    assert "mapbox.com/about/maps" in html and "openstreetmap.org/copyright" in html
    assert re.search(r"\btileSize\"?: 512\b", html)
    assert re.search(r"\bzoomOffset\"?: -1\b", html)


@pytest.mark.parametrize("tiles", ["OpenStreetMap", "Mapbox"])
@pytest.mark.parametrize("API_key", [None, ""])
def test_plot_map_without_api_key(tiles, API_key):
    # As before: no key is fine for OpenStreetMap, and an error for Mapbox
    model, _, _ = _fitted("infostop")
    if tiles == "Mapbox":
        with pytest.raises(ValueError):
            plot_map(model, tiles=tiles, API_key=API_key)
    else:
        assert isinstance(plot_map(model, tiles=tiles, API_key=API_key).m, folium.Map)


@pytest.mark.parametrize(
    "tiles",
    [
        "OpenStreetMap",
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        None,
        "provider without a key",
        "provider with two keys",
    ],
)
def test_plot_map_api_key_unused_by_other_tiles(tiles):
    # Tiles that need no key, or more than one, give the same map (or error) with
    # and without a key
    if tiles == "provider without a key":
        xyzservices = pytest.importorskip("xyzservices")
        # 512 px tiles: folium itself drops these options, so the page shows which
        # code built the tile layer
        tiles = xyzservices.TileProvider(
            name="Test.NoKey",
            url="https://tiles.example.com/{z}/{x}/{y}.png",
            attribution="(C) Test",
            tileSize=512,
            zoomOffset=-1,
        )
    elif tiles == "provider with two keys":
        xyzservices = pytest.importorskip("xyzservices")
        # Like the old HERE tiles in xyzservices <= 2025.4
        tiles = xyzservices.TileProvider(
            name="Test.TwoKeys",
            url="https://tiles.example.com/{z}/{x}/{y}.png?id={app_id}&code={app_code}",
            attribution="(C) Test",
            app_id="<insert your app_id here>",
            app_code="<insert your app_code here>",
        )
    model, _, _ = _fitted("infostop")
    results = []
    for API_key in [None, "my-token"]:
        try:
            html = plot_map(model, tiles=tiles, API_key=API_key).m.get_root().render()
        except Exception as error:  # folium < 0.14 takes no provider objects
            results.append(type(error))
            continue
        assert "my-token" not in html
        # folium gives each element a random id
        results.append(re.sub(r"_[0-9a-f]{32}", "_ID", html))
    assert results[0] == results[1]


# Tile sets that need a key, and the field of the provider that the key goes into
KEYED_TILES = [
    ("Thunderforest.OpenCycleMap", "apikey"),
    ("MapTiler.Streets", "key"),
    ("Jawg.Streets", "accessToken"),
    ("TomTom.Basic", "apikey"),
    ("CartoDB positron", "apikey"),
]


def _render_with_layer_control(tiles, API_key):
    model, _, _ = _fitted("infostop")
    folmap = plot_map(model, tiles=tiles, API_key=API_key)
    # The layer control shows the name of the tile layer
    folium.LayerControl().add_to(folmap.m)
    return folmap.m.get_root().render()


@pytest.mark.skipif(OLD_FOLIUM, reason="folium < 0.9 does not know these tiles")
@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("tiles, field", KEYED_TILES)
def test_plot_map_api_key_for_tiles_that_need_one(tiles, field, as_object):
    xyzservices = pytest.importorskip("xyzservices")
    provider = xyzservices.providers.query_name(tiles)
    if not provider.requires_token():
        pytest.skip(f"{tiles} needs no key in xyzservices {xyzservices.__version__}")
    html = _render_with_layer_control(provider if as_object else tiles, "my-token")
    tile_urls = re.findall(r"L\.tileLayer\(\s*[\"']([^\"']+)", html)
    url = provider(**{field: "my-token"}).build_url(
        fill_subdomain=False, scale_factor="{r}"
    )
    assert tile_urls == [url]
    # The key is only in the tile URL
    assert html.count("my-token") == 1
    # The layer control lists the layer under the name folium gives provider layers
    assert re.search(rf"\"{provider.name.replace('.', '').lower()}\"\s*:", html)
    links = re.findall(r'href="([^"]+)"', provider.html_attribution)
    assert links and all(link in html for link in links)
    # The tile options of the provider
    assert re.search(rf"\bmaxZoom\"?: {provider['max_zoom']}\b", html)
    if "subdomains" in provider:
        assert re.search(rf"\bsubdomains\"?: \"{provider['subdomains']}\"", html)
    if "tileSize" in provider:
        assert re.search(rf"\btileSize\"?: {provider['tileSize']}\b", html)
        assert re.search(rf"\bzoomOffset\"?: {provider['zoomOffset']}\b", html)


@pytest.mark.skipif(OLD_FOLIUM, reason="folium < 0.9 does not use the key here")
def test_plot_map_api_key_with_every_tile_option():
    xyzservices = pytest.importorskip("xyzservices")
    provider = xyzservices.TileProvider(
        name="Test.Keyed",
        url="https://{s}.tiles.example.com/{z}/{x}/{y}{r}.png?token={token}",
        attribution="(C) Test",
        token="<insert your token here>",
        min_zoom=3,
        max_zoom=15,
        subdomains="pq",
        tileSize=512,
        zoomOffset=-1,
    )
    html = _render_with_layer_control(provider, "my-token")
    tile_urls = re.findall(r"L\.tileLayer\(\s*[\"']([^\"']+)", html)
    assert tile_urls == [
        "https://{s}.tiles.example.com/{z}/{x}/{y}{r}.png?token=my-token"
    ]
    assert html.count("my-token") == 1
    assert re.search(r"\"testkeyed\"\s*:", html)
    assert "(C) Test" in html
    for option, value in [
        ("minZoom", 3),
        ("maxZoom", 15),
        ("subdomains", '"pq"'),
        ("tileSize", 512),
        ("zoomOffset", -1),
    ]:
        assert re.search(rf"\b{option}\"?: {value}", html), option


@pytest.mark.skipif(not OLD_FOLIUM, reason="folium >= 0.9 uses the key")
def test_plot_map_api_key_on_old_folium():
    pytest.importorskip("xyzservices")
    model, _, _ = _fitted("infostop")
    # Tiles that need a key and that folium 0.7 / 0.8 do not know
    with pytest.raises(ValueError, match="only on folium 0.9 and newer"):
        plot_map(model, tiles="Thunderforest.OpenCycleMap", API_key="my-token")
    # Other errors stay folium's own
    for tiles, API_key in [
        ("Thunderforest.OpenCycleMap", None),
        ("OpenStreetMapp", "k"),
    ]:
        with pytest.raises(ValueError) as error:
            plot_map(model, tiles=tiles, API_key=API_key)
        assert "API_key" not in str(error.value)
    # folium 0.7 / 0.8 have CartoDB tiles of their own, which get no key, as before
    html = plot_map(model, tiles="CartoDB positron", API_key="my-token").m.get_root()
    assert "my-token" not in html.render()


def test_plot_map_passes_plain_floats_to_folium(monkeypatch):
    # folium 0.12-0.13 cannot take a numpy array as a marker location, and folium
    # 0.7-0.8 write numpy 2 floats into the page as `np.float64(...)`
    passed = []
    circle_marker, heatmap = folium.CircleMarker, visualize.HeatMap
    monkeypatch.setattr(
        folium,
        "CircleMarker",
        lambda location, **kw: passed.append(location) or circle_marker(location, **kw),
    )
    monkeypatch.setattr(
        visualize,
        "HeatMap",
        lambda data, **kw: passed.extend(data) or heatmap(data, **kw),
    )
    model, _, _ = _fitted("infostop")
    folmap = plot_map(model, display_data="all", scatter=True)
    # First the markers, then the heatmap, each at the shown points
    assert passed == 2 * folmap.points.tolist()
    assert all(type(value) is float for location in passed for value in location)
    assert "np.float64" not in folmap.m.get_root().render()


def test_folium_map_with_points_as_a_list():
    points = [[55.70, 12.50], [55.71, 12.52], [55.70, 12.51]]
    folmap = FoliumMap(points, np.array([0, 1, 0]))
    folmap.render_points(subsampling=0.5)
    folmap.render_heatmap(subsampling=0.5)
    assert "<html>" in folmap.m.get_root().render()
