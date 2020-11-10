import folium
from folium.plugins import HeatMap
import numpy as np
from infostop import utils

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
            Default is 'OpenStreetMap'. A good minimalist alternative is 'CartoDBpositron'.
        API_key : str
            Mapbox API key.
    
    Example
    -------
        >>> folmap = FoliumMap(mypoints, mylabels)
        >>> folmap.render_polygons()
        >>> folmap.render_heatmap()
        >>> folmap.m
    """
    def __init__(self, points, labels, zoom_start=10, tiles='OpenStreetMap', API_key=None):
        self.points = points
        self.labels = labels
        self.m = folium.Map(
            location=tuple(np.median(points, axis=0)),
            zoom_start=zoom_start,
            tiles=tiles,
            API_key=API_key,
            attr="Infostop"
        )


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
            return {"fillColor": feature["properties"]["color"], "color": feature["properties"]["color"], "weight": 1, "fillOpacity": opacity}

        stop_hulls = []
        for stop_idx in (set(self.labels) - {-1}):
            p = self.points[self.labels == stop_idx]
            stop_hulls.append(utils.convex_hull(p))

        features = {"type": "FeatureCollection", "features": [
            {
                "type": "Feature",
                "properties": {"color": color},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": []
                }
            }
        ]}

        for hull in stop_hulls:
            features["features"][0]["geometry"]["coordinates"].append(
                hull[:, ::-1].tolist()
            )

        folium.GeoJson(
            features,
            style_function=_style_function
        ).add_to(self.m)


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
        p = self.points
        if subsampling < 1:
            mask = np.random.choice(range(p.shape[0]), size=int(subsampling * p.shape[0]), replace=False)
            p = p[mask]

        for d in p:
            folium.CircleMarker(
                d[:2],
                color=color,
                fill_color=color,
                fill_opacity=opacity,
                fill=True
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
        p = self.points
        if subsampling < 1:
            mask = np.random.choice(range(p.shape[0]), size=int(subsampling * p.shape[0]), replace=False)
            p = p[mask]
        HeatMap(p, radius=radius).add_to(self.m)


def plot_map(model, display_data="unique_stationary", polygons=True, scatter=False, heatmap=True, polygons_color="#ee9999", polygons_opacity=0.3, scatter_color="k", scatter_opacity=0.3, scatter_subsampling=1, heatmap_radius=8, heatmap_subsampling=1, zoom_start=12, tiles='OpenStreetMap', API_key=None):
    """Load Folium map to display the data and fit.

    Note: `plot_map` only works if `distance_metric` is "havesine".

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
            Default is 'OpenStreetMap'. A good minimalist alternative is 'CartoDBpositron'.
        API_key : str
            Mapbox API key.

    Returns
    -------
        folmap : folium.folium.Map

    Example
    -------
        >>> # create model and fit it to data
        >>> model = Infostop()
        >>> _ = model._fit_predict(traces)
        >>> # visualize result
        >>> folmap = visualize.plot_map(model)
        >>> folmap.m
    """
    assert hasattr(model, "_is_fitted"), \
           "`model` must either be an instance of `Infostop` or `SpatialInfomap`."
    assert model._is_fitted, \
           "It appears that no data is fitted to the model."
    assert model._distance_metric == "haversine", \
           "`distance_metric` is not 'haversine'."
    assert display_data in ['all', 'all_stationary', 'all_nonstationary', 'unique_stationary'], \
           'Keyword argument `display_data` must be one of "all", "all_stationary", "all_nonstationary" and "unique_stationary".'

    # If it's an instance of Infostop
    if hasattr(model, "r1"):
        if "all" in display_data:
            data_unique, indices_unique = np.unique(np.vstack(model._data)[:, :2], axis=0, return_index=True)
            labels_unique = np.hstack(model._coord_labels)[indices_unique]
            if display_data == "all":
                points = data_unique
                labels = labels_unique
            elif display_data == "all_stationary":
                points = data_unique[labels_unique>=0]
                labels = labels_unique[labels_unique>=0]
            elif display_data == "all_nonstationary":
                points = data_unique[labels_unique==-1]
                labels = labels_unique[labels_unique==-1]
        elif display_data == "unique_stationary":
            points = model._stat_coords[model._stat_labels >= 0]
            labels = model._stat_labels[model._stat_labels >= 0]

    # If it's an instance of SpatialInfomap
    else:
        points = model._stat_coords[model._stat_labels >= 0]
        labels = model._stat_labels[model._stat_labels >= 0]

    folmap = FoliumMap(
        points=points,
        labels=labels,
        zoom_start=zoom_start,
        tiles=tiles,
        API_key=API_key
    )

    if polygons:
        folmap.render_polygons(color=polygons_color, opacity=polygons_opacity)
    if scatter:
        folmap.render_points(color=scatter_color, opacity=scatter_opacity, subsampling=scatter_subsampling)
    if heatmap:
        folmap.render_heatmap(radius=heatmap_radius, subsampling=heatmap_subsampling)

    return folmap