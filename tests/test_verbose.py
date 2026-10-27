import re

import numpy as np
import pytest

from infostop import Infostop
from infostop.models import SpatialInfomap

HOME, WORK = (55.70, 12.50), (55.71, 12.52)


def make_trace(places, seed=0, n=30):
    """One Gaussian blob of `n` points per visit, 60 s apart, visits 4000 s apart."""
    rng = np.random.default_rng(seed)
    visits = []
    for i, (lat, lon) in enumerate(places):
        visits.append(
            np.column_stack(
                [
                    lat + rng.normal(0, 2e-5, n),
                    lon + rng.normal(0, 2e-5, n),
                    4000 * i + 60 * np.arange(n),
                ]
            )
        )
    return np.vstack(visits)


def printed_count(capsys):
    # Not anchored at line start: with an empty network, the line before ends
    # without a newline ("... running Infomap... Found 0 stop locations").
    counts = re.findall(r"Found (\d+) stop locations$", capsys.readouterr().out, re.M)
    assert len(counts) == 1
    return int(counts[0])


@pytest.mark.parametrize("label_singleton", [True, False])
@pytest.mark.parametrize(
    "places, expected",
    [
        ([HOME, WORK, HOME], 1),  # home revisited, work a singleton
        ([HOME, WORK, HOME, WORK], 2),  # both revisited
        ([HOME, WORK], 0),  # nothing revisited: the Infomap network is empty
    ],
)
def test_infostop_verbose_count(capsys, places, expected, label_singleton):
    Infostop(verbose=True, label_singleton=label_singleton).fit_predict(
        make_trace(places)
    )
    assert printed_count(capsys) == expected


@pytest.mark.parametrize("seed", range(5))
def test_infostop_verbose_count_matches_labels(capsys, seed):
    # With `label_singleton=False`, every label other than -1 is an Infomap module.
    rng = np.random.default_rng(seed)
    places = [(55.70 + 0.01 * i, 12.50) for i in rng.integers(0, 4, size=12)]
    model = Infostop(verbose=True, label_singleton=False)
    labels = model.fit_predict(make_trace(places, seed=seed))
    assert printed_count(capsys) == len(set(labels) - {-1})


def test_spatial_infomap_verbose_count(capsys):
    coords = make_trace([HOME, WORK])[:, :2]
    labels = SpatialInfomap(verbose=True, label_singleton=False).fit_predict(coords)
    assert len(set(labels) - {-1}) == 2
    assert printed_count(capsys) == 2
