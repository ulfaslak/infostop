# Changelog

## 0.2.0 (2026-10-28)

This is the first release since 0.1.9 (February 2022). This entry lists everything that changed since 0.1.9. Versions 0.1.10 and 0.1.11 were never published. The git tags `v1.0.10` and `v1.0.11` are not releases either: they mark two commits from 2022 whose code is the same as 0.1.9.

### Installs on current Python and current Infomap

- **Infomap.** infostop 0.1.9 required exactly `infomap==1.0.6`, which no longer installs on current Python. infostop now requires `infomap>=1.0.6` and uses the API of current Infomap. It works with Infomap 1.0.6 to 2.16. ([#56](https://github.com/ulfaslak/infostop/pull/56))
- **Wheels.** PyPI now has infostop wheels for Python 3.9 to 3.14 on Linux (x86_64 and aarch64), macOS (Intel and Apple Silicon) and Windows (x86_64). With a wheel, pip does not compile infostop, so you do not need a C++ compiler for infostop. 0.1.9 had a wheel only for Python 3.8 on Intel Macs. ([#63](https://github.com/ulfaslak/infostop/pull/63))
- **Infomap may still need a compiler.** Current Infomap has wheels only for Python 3.11 and later. On Python 3.9 and 3.10, pip compiles Infomap 2.9.2, the last Infomap for these versions, and this needs a C++ compiler. On Intel Macs, and on Apple Silicon Macs older than macOS 15, Infomap has no wheels either: install it with `OPENMP=0 pip install infomap` first, as README "Installation" shows. ([#78](https://github.com/ulfaslak/infostop/pull/78))
- **Windows.** The C++ step now compiles with Microsoft's compiler (MSVC). Before, the build failed with `'M_PI': undeclared identifier`. ([#26](https://github.com/ulfaslak/infostop/pull/26))
- **Other systems** get the source package (sdist) and compile it, as before. This includes Alpine Linux, and Python versions without a wheel.
- **SciPy** is now a declared dependency. infostop already used it. It now imports `QhullError` from `scipy.spatial`, because SciPy 2.0 removes the private module `scipy.spatial.qhull`. ([#65](https://github.com/ulfaslak/infostop/pull/65))
- **No `tests` package.** The 0.1.9 package also installed a top-level package named `tests` into your environment. It no longer does. ([#70](https://github.com/ulfaslak/infostop/pull/70))
- **Upgrading keeps your Infomap.** `pip install -U infostop` keeps an installed Infomap 1.0.6, because it meets `infomap>=1.0.6`. A new environment gets the newest Infomap for its Python. Newer Infomap versions can put points in crowded places into different stops (see the next section). To reproduce results from 0.1.9 exactly, pin `infostop==0.1.9` and `infomap==1.0.6`, and use the same operating system.

### Labels on crowded stops change once: new `num_trials` parameter, default 50

`Infostop` and `SpatialInfomap` have a new parameter, `num_trials`. Infomap now clusters the network 50 times from different random starts and keeps the partition with the shortest codelength. Before, it ran once. The measurements behind the numbers below are in [PR #79](https://github.com/ulfaslak/infostop/pull/79).

- **Why.** With a single run, the labels of crowded stops depended on the Infomap version and on the operating system. Crowded stops are stops that overlap within `r2`. In our tests, about 1 in 6 crowded traces got different labels between Infomap 1.0.6 and 2.13, and between macOS and Linux. With 50 runs they agreed.
- **What changes for you.**
  - Labels of clearly separated stops do not change, and points labelled `-1` (moving) never change.
  - On some crowded stops, labels change once: in our tests, 20 of 300 crowded traces got different labels than infostop 0.1.9 gave with Infomap 1.0.6. Each time the new partition has a shorter codelength, so it is a better solution by Infomap's own measure. On realistic multi-day traces we saw no change.
- **Label numbers.** Stops that Infomap groups are numbered by flow, largest first. Infomap 2.9 to 2.15 number them in a different order when they run more than one trial, so infostop renumbers them by flow. Where Infomap already numbers by flow, nothing changes.
- **Caveat on ties.** Some stops have exactly the same flow. On Infomap 2.9 to 2.15 (what Python 3.9 and 3.10 users get), such stops can be numbered differently from infostop 0.1.9 with 50 trials. The order of exactly tied stops also already differed between macOS and Linux before this change. Which points share a label is not affected.
- **The old behaviour.** `Infostop(num_trials=1)` and `SpatialInfomap(num_trials=1)` run Infomap once, as before. With 1 run, labels on crowded stops again depend on the Infomap version and the operating system.
- **Cost.** The clustering step runs 50 times. In our tests, `fit_predict` on 33,000 points took 0.36 s instead of 0.05 s, and on 340,000 points 4.0 s instead of 0.6 s.

### Other changes to labels

- **`label_singleton=False`: a place visited several times at exactly one coordinate now gets a label.** Before, it got `-1`, as if it was visited once. This happens when all stop events at a place have the same median coordinate: when `min_spacial_resolution` rounds them together, or when the input coordinates are already rounded or snapped to fixed points (for example WiFi or cell positions). A place with only one stop event still gets `-1`. With the default `label_singleton=True`, nothing changes. This applies to `Infostop` and `SpatialInfomap`. It is the only change in this release that gives a label to points that were `-1`. ([#68](https://github.com/ulfaslak/infostop/pull/68))
- **NumPy 2.0.0.** With NumPy 2.0.0 (only that version), `Infostop.fit_predict` failed with a `ValueError`, and `SpatialInfomap.fit_predict` returned labels with shape `(N, 1)` instead of `(N,)`. Both now return one label per input row. ([#73](https://github.com/ulfaslak/infostop/pull/73))

### `compute_intervals` gives one interval per stay

`infostop.postprocess.compute_intervals` had three bugs. ([#53](https://github.com/ulfaslak/infostop/pull/53))

- When a trace ended at a stop, the last stop interval was missing. Now it is there. A single point now gives one interval.
- Two points with the same label and a time gap of exactly `max_time_between` gave two intervals. Now they give one. This is the rule `Infostop` uses to make stop events.
- An empty input raised `IndexError`. Now it returns `[]`.

### Input that now works, and clearer errors

- **A tuple of traces** works like a list, and returns a tuple of label arrays. Before, it failed with an `AttributeError` or a `ValueError`. ([#42](https://github.com/ulfaslak/infostop/pull/42))
- **Users with and without timestamps.** In a list of traces, some traces can have a time column and some not. Before, this failed with a `ValueError` from NumPy. ([#55](https://github.com/ulfaslak/infostop/pull/55))
- **A user with no points** gets an empty label array. The other users get the same labels as without that user. Before, this failed with a `ValueError` (haversine), or read past the end of the array in the C++ step (euclidean). ([#57](https://github.com/ulfaslak/infostop/pull/57), [#74](https://github.com/ulfaslak/infostop/pull/74))
- **Clearer errors.** These inputs now raise an `AssertionError` that says what is wrong. Before, most of them failed inside NumPy or the C++ step with another error:
  - an empty list: "There are no traces in the input data."; only empty traces: "There are no points in the input data." ([#55](https://github.com/ulfaslak/infostop/pull/55), [#57](https://github.com/ulfaslak/infostop/pull/57))
  - a trace that is not a 2D array, for example a single trace given as a list of rows: "Each trace must be a 2D numpy array with one row per point ..." ([#42](https://github.com/ulfaslak/infostop/pull/42))
  - NaN values: the message now gives the number of NaN values, not the whole mask. ([#39](https://github.com/ulfaslak/infostop/pull/39))
  - `num_trials` that is not an integer of 1 or more. This check also runs under `python -O`. ([#79](https://github.com/ulfaslak/infostop/pull/79))
- **The C++ step checks its arguments.** `cpputils.get_stationary_events` raises `ValueError` for a `distance_metric` other than `"haversine"` or `"euclidean"`, and for coordinates with other than 2 or 3 columns. Before, it gave wrong results without an error. `Infostop` already checked both, so this changes nothing for `Infostop` unless you run Python with `-O`, which turns off its checks. ([#75](https://github.com/ulfaslak/infostop/pull/75))

### `plot_map`

- **`display_data` works.** Before, every option showed the stop-event medians, which is the default `"unique_stationary"`. Now `"all"`, `"all_stationary"` and `"all_nonstationary"` show the input points they name, for `Infostop` and `SpatialInfomap`. The default output does not change. ([#43](https://github.com/ulfaslak/infostop/pull/43))
- **Nothing to show.** When no point matches `display_data` (for example, when every label is `-1`), `plot_map` returns an empty map. Before, it raised `ValueError: Location values cannot contain NaNs`. `FoliumMap` has a new optional `location` parameter for the map center. ([#60](https://github.com/ulfaslak/infostop/pull/60))
- **Tile credits.** Maps credit their tiles again, for example "© OpenStreetMap contributors". Before, the credit said "Infostop" on folium 0.7, 0.8, and 0.15 and later. ([#66](https://github.com/ulfaslak/infostop/pull/66))
- **Mapbox.** `tiles="Mapbox"` with `API_key` works again on folium 0.9 and later. Before, folium raised an error. It uses the `xyzservices` package, which folium 0.15.1 and later install. Also, on folium 0.9 and later, the key was written into the map's HTML with any tiles. Now it goes only into the Mapbox tile URL. ([#69](https://github.com/ulfaslak/infostop/pull/69))
- **Older folium.** Scatter markers work on folium 0.12 and 0.13, and the heatmap shows on folium 0.7 and 0.8 with NumPy 2. ([#71](https://github.com/ulfaslak/infostop/pull/71))

### Other fixes

- With `verbose=True`, "Found N stop locations" was one too low. ([#41](https://github.com/ulfaslak/infostop/pull/41))
- A model that was pickled with infostop 0.1.9 runs Infomap once (`num_trials=1`) when you call `fit_predict` on it again. ([#79](https://github.com/ulfaslak/infostop/pull/79))

### Documentation

- The documentation on [Read the Docs](https://infostop.readthedocs.io) is rewritten for the current API, with examples that are tested. ([#44](https://github.com/ulfaslak/infostop/pull/44))
- The docstrings give the units of each parameter: with `distance_metric="haversine"`, `r1` and `r2` are in meters; the time parameters are in the units of the time column, and their defaults assume seconds. `max_time_between` was described as the longest duration of a stop. It is the longest time between two consecutive points in a stop; the code did not change. ([#52](https://github.com/ulfaslak/infostop/pull/52))
