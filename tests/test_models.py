from infostop import Infostop

import numpy as np


# Example test function. Run `make test` to see that it passes.`
def test_example():
    # Function
    func = _.area

    # Params and returns
    params = dict(polygon=[[1, 0], [0, 0], [0, 1], [1, 1]])
    returns = 1

    # Test
    assert func(**params) == returns
