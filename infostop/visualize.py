import inspect
from urllib.parse import urlsplit
import warnings

import folium
from folium.plugins import HeatMap
import numpy as np
from infostop import utils

# Tile options of an `xyzservices` provider that the tile layer gets. folium uses the
# first three for providers itself; it drops the last two (512 px tiles, for example
# Mapbox and MapTiler).
_TILE_OPTIONS = ["min_zoom", "max_zoom", "subdomains", "tileSize", "zoomOffset"]


def _keyed_provider(tiles):
    """Return the `xyzservices` provider of `tiles` and its key field, if it needs a key.

    `xyzservices` is installed with folium >= 0.15.1. It marks the key field of these
    tiles with a "<insert your ... here>" placeholder. Return None if `tiles` does not
    name (or is not) a provider that needs one key, or if `xyzservices` is not
    installed.
    """
    try:
        import xyzservices
    except ImportError:
        return None
    if isinstance(tiles, str):
        try:
            provider = xyzservices.providers.query_name(tiles)
        except ValueError:
            return None
    elif isinstance(tiles, xyzservices.TileProvider):
        provider = tiles
    else:
        return None
    if not provider.requires_token():
        return None
    # The same test as `requires_token`: the field has a placeholder and is in the URL
    fields = [
        field
        for field, value in provider.items()
        if isinstance(value, str) and "<insert your" in value and field in provider.url
    ]
    # One key only: the old HERE tiles need two (app_id and app_code)
    return (provider, fields[0]) if len(fields) == 1 else None


def _keyed_tiles(tiles, API_key, attr=None):
    """Return a tile layer with `API_key` in its URL, for tiles that need a key.

    Return None if `tiles` does not need one key (see `_keyed_provider`).
    """
    keyed = _keyed_provider(tiles)
    if keyed is None:
        return None
    provider, field = keyed
    provider = provider(**{field: API_key})
    return folium.TileLayer(
        tiles=provider.build_url(fill_subdomain=False, scale_factor="{r}"),
        attr=attr or provider.html_attribution,
        # folium names layers of providers like this; not after the URL with the key
        name=provider.name.replace(".", "").lower(),
        **{option: provider[option] for option in _TILE_OPTIONS if option in provider},
    )


def _url_credit(url):
    """Return the credit for a custom tile URL without `attr`: the host of the URL.

    folium requires a credit for a custom URL. The host names where the tiles come
    from without claiming to be their author.
    """
    # No user name or password, and no "{s}" subdomain placeholder
    host = urlsplit(url).netloc.rpartition("@")[2].replace("{s}.", "")
    return f"Tiles from {host}" if host else "Custom tiles"


class FoliumMap:
    """Plot points on a folium map.

    Parameters
    ----------
        points : 2d numpy.array (shape=(N, 2))
        labels : 1d numpy.array (shape=(N, ))
        zoom_start : int/float
            Initial zoom of the map.
        tiles : str
            Which background tiles to use in the map. Any valid tiles for `folium.Map` method.
            Default is 'OpenStreetMap'.
        API_key : str
            API key for tiles that need one, such as 'Mapbox'. It goes into the tile
            URL. It works for the tile sets in the `xyzservices` package that need a
            key. This needs folium 0.9 or newer with `xyzservices` installed (folium
            0.15.1 and newer installs it). On folium 0.7 and 0.8, only 'Mapbox' works.
        location : tuple of floats
            Center of the map. Default is the median of `points`, so it must be given
            if `points` is empty.
        attr : str
            Credit for the tiles, shown in the corner of the map. Give it when `tiles`
            is a tile URL, for example `attr="© OpenTopoMap (CC-BY-SA)"`. By default,
            named tiles show their own credit, and a tile URL shows
            "Tiles from <host>". folium 0.9 to 0.14 always show the own credit of
            named tiles.

    Example
    -------
        >>> folmap = FoliumMap(mypoints, mylabels)
        >>> folmap.render_polygons()
        >>> folmap.render_heatmap()
        >>> folmap.m
    """

    def __init__(
        self,
        points,
        labels,
        zoom_start=10,
        tiles="OpenStreetMap",
        API_key=None,
        location=None,
        attr=None,
    ):
        self.points = points
        self.labels = labels
        if location is None:
            location = tuple(np.median(points, axis=0))
        map_kwargs = dict(location=location, zoom_start=zoom_start, tiles=tiles)
        old_folium = "API_key" in inspect.signature(folium.Map).parameters
        keyed = None
        if old_folium:
            # folium 0.7 and 0.8 put the key in the URL of their own Mapbox tiles
            map_kwargs["API_key"] = API_key
        elif API_key:
            keyed = _keyed_tiles(tiles, API_key, attr)
        if keyed is not None:
            # folium cannot put the key into the URL of these tiles itself
            map_kwargs["tiles"] = None
        elif isinstance(tiles, str) and "{" in tiles:
            # folium requires an attribution for a custom tile URL
            map_kwargs["attr"] = attr or _url_credit(tiles)
        elif isinstance(tiles, str) and old_folium:
            # folium < 0.9 shows the attribution of named tiles only if it is passed
            try:
                tiles_attr = folium.TileLayer(tiles=tiles, API_key=API_key).attr
            except ValueError as error:
                # folium < 0.9 does not know these tiles
                if API_key and _keyed_provider(tiles) is not None:
                    raise ValueError(
                        f"API_key works with tiles={tiles!r} only on folium 0.9 and "
                        f"newer. This is folium {folium.__version__}."
                    ) from error
                raise
            map_kwargs["attr"] = attr or tiles_attr
        elif attr:
            if isinstance(tiles, str):
                # With an attribution, folium takes an unknown tile name for a tile
                # URL and shows no tiles. Without one, it raises for an unknown name.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    folium.TileLayer(tiles=tiles)
            map_kwargs["attr"] = attr
        self.m = folium.Map(**map_kwargs)
        if keyed is not None:
            keyed.add_to(self.m)

    def render_polygons(self, color="#ee9999", opacity=0.3):
        """Render convex hulls of points in each label-group.

        Parameters
        ----------
            color : str
                Color of stop location areas.
            opacity : float (in [0, 1])
                Opacity of stop location areas.
        """

        def _style_function(feature):
            return {
                "fillColor": feature["properties"]["color"],
                "color": feature["properties"]["color"],
                "weight": 1,
                "fillOpacity": opacity,
            }

        stop_hulls = []
        for stop_idx in set(self.labels) - {-1}:
            p = self.points[self.labels == stop_idx]
            stop_hulls.append(utils.convex_hull(p))

        features = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"color": color},
                    "geometry": {"type": "Polygon", "coordinates": []},
                }
            ],
        }

        for hull in stop_hulls:
            features["features"][0]["geometry"]["coordinates"].append(
                hull[:, ::-1].tolist()
            )

        folium.GeoJson(features, style_function=_style_function).add_to(self.m)

    def render_points(self, color="k", opacity=0.3, subsampling=1):
        """Render points.

        Parameters
        ----------
            color : str
                Color of points.
            opacity : float (in [0, 1])
                Opacity of points.
            subsampling : float (in [0, 1])
                Sampling rate of scattered points. For example, using 0.2 shows only 20 percent
                of the data.
        """
        p = np.asarray(self.points)
        if subsampling < 1 and len(p) > 0:
            mask = np.random.choice(
                range(p.shape[0]), size=int(subsampling * p.shape[0]), replace=False
            )
            p = p[mask]

        for d in p:
            # Plain floats: some folium versions cannot take numpy values
            folium.CircleMarker(
                d[:2].tolist(),
                color=color,
                fill_color=color,
                fill_opacity=opacity,
                fill=True,
            ).add_to(self.m)

    def render_heatmap(self, radius=8, subsampling=1):
        """Render points.

        Parameters
        ----------
            radius : int/float
                Kernel radius of points rendered in the heatmap. It makes sense to adjust inversely
                to the density of points, but in general [8, 10] gives good results.
            subsampling : float (in [0, 1])
                Sampling rate of points used for rendering the heatmap. For example, using 0.2
                renders a heatmap based on only 20 percent of the data.
        """
        p = np.asarray(self.points)
        if subsampling < 1 and len(p) > 0:
            mask = np.random.choice(
                range(p.shape[0]), size=int(subsampling * p.shape[0]), replace=False
            )
            p = p[mask]
        # Plain floats: some folium versions cannot take numpy values
        HeatMap(p.tolist(), radius=radius).add_to(self.m)


def plot_map(
    model,
    display_data="unique_stationary",
    polygons=True,
    scatter=False,
    heatmap=True,
    polygons_color="#ee9999",
    polygons_opacity=0.3,
    scatter_color="k",
    scatter_opacity=0.3,
    scatter_subsampling=1,
    heatmap_radius=8,
    heatmap_subsampling=1,
    zoom_start=12,
    tiles="OpenStreetMap",
    API_key=None,
    attr=None,
):
    """Load Folium map to display the data and fit.

    Note: `plot_map` only works if `distance_metric` is "haversine".

    Parameters
    ----------
        model : fitted instance of `Infostop` or `SpatialInfomap`
        display_data : str
            Which data to display. Defaults to "unique_stationary" which is the downsampled data
            that Infostop is fitted on (the medians of all stationary location events, where
            duplicates have been removed). Other options are "all", which displays all of the
            original input data, "all_stationary" which displays all the input data except points
            that are labeled as non-stationary, and "all_nonstationary" which displays all the
            non-stationary labeled input data.
        polygons : bool
            Display the convex hull polygons representing fitted stop locations.
        scatter : bool
            Display points (slow: not recommended for over 10.000 points).
        heatmap : bool
            Display a heatmp (typically a better option than `scatter==True`).
        polygons_color : str
            Color of stop location areas.
        polygons_opacity : float (in [0, 1])
            Opacity of stop location areas.
        scatter_color : str
            Color of points.
        scatter_opacity : float (in [0, 1])
            Opacity of points.
        scatter_subsampling : float (in [0, 1])
            Sampling rate of scattered points. For example, using 0.2 shows only 20 percent
            of the data.
        heatmap_radius : int/float
            Kernel radius of points rendered in the heatmap. It makes sense to adjust inversely
            to the density of points, but in general [8, 10] gives good results.
        heatmap_subsampling : float (in [0, 1])
            Sampling rate of points used for rendering the heatmap. For example, using 0.2
            renders a heatmap based on only 20 percent of the data.
        zoom_start : int/float
            Initial zoom of the map.
        tiles : str
            Which background tiles to use in the map. Any valid tiles for `folium.Map` method.
            Default is 'OpenStreetMap'.
        API_key : str
            API key for tiles that need one, such as 'Mapbox'. It goes into the tile
            URL. It works for the tile sets in the `xyzservices` package that need a
            key. This needs folium 0.9 or newer with `xyzservices` installed (folium
            0.15.1 and newer installs it). On folium 0.7 and 0.8, only 'Mapbox' works.
        attr : str
            Credit for the tiles, shown in the corner of the map. Give it when `tiles`
            is a tile URL, for example `attr="© OpenTopoMap (CC-BY-SA)"`. By default,
            named tiles show their own credit, and a tile URL shows
            "Tiles from <host>". folium 0.9 to 0.14 always show the own credit of
            named tiles.

    Returns
    -------
        folmap : FoliumMap
            The folium map is `folmap.m`.

    Example
    -------
        >>> # create model and fit it to data
        >>> model = Infostop()
        >>> _ = model.fit_predict(traces)
        >>> # visualize result
        >>> folmap = plot_map(model)
        >>> folmap.m
    """
    assert hasattr(
        model, "_is_fitted"
    ), "`model` must either be an instance of `Infostop` or `SpatialInfomap`."
    assert model._is_fitted, "It appears that no data is fitted to the model."
    assert (
        model._distance_metric == "haversine"
    ), "`distance_metric` is not 'haversine'."
    assert display_data in [
        "all",
        "all_stationary",
        "all_nonstationary",
        "unique_stationary",
    ], 'Keyword argument `display_data` must be one of "all", "all_stationary", "all_nonstationary" and "unique_stationary".'

    if display_data == "unique_stationary":
        points = model._stat_coords[model._stat_labels >= 0]
        labels = model._stat_labels[model._stat_labels >= 0]
    else:
        # If it's an instance of Infostop
        if hasattr(model, "_r1"):
            coords = np.vstack([coords_u[:, :2] for coords_u in model._data])
            coord_labels = np.hstack(model.labels)

        # If it's an instance of SpatialInfomap
        else:
            coords = model._data
            coord_labels = model._labels

        # Remove duplicate points, but keep a point once for every label it has
        unique = np.unique(np.column_stack([coords, coord_labels]), axis=0)
        data_unique, labels_unique = unique[:, :2], unique[:, 2].astype(int)
        if display_data == "all":
            points = data_unique
            labels = labels_unique
        elif display_data == "all_stationary":
            points = data_unique[labels_unique >= 0]
            labels = labels_unique[labels_unique >= 0]
        elif display_data == "all_nonstationary":
            points = data_unique[labels_unique == -1]
            labels = labels_unique[labels_unique == -1]

    # With nothing to show, center the map on the coordinates the model clustered
    location = None
    if len(points) == 0:
        location = tuple(np.median(model._stat_coords, axis=0))

    folmap = FoliumMap(
        points=points,
        labels=labels,
        zoom_start=zoom_start,
        tiles=tiles,
        API_key=API_key,
        location=location,
        attr=attr,
    )

    if polygons:
        folmap.render_polygons(color=polygons_color, opacity=polygons_opacity)
    if scatter:
        folmap.render_points(
            color=scatter_color,
            opacity=scatter_opacity,
            subsampling=scatter_subsampling,
        )
    if heatmap:
        folmap.render_heatmap(radius=heatmap_radius, subsampling=heatmap_subsampling)

    return folmap
