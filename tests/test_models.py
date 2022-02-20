from infostop import Infostop

import numpy as np


# Example test function. Run `make test` to see that it passes.`
def test_example():
    # Function
    func = lambda x: f"I'm an example {x}"

    # Params and returns
    params = dict(x="function")
    returns = "I'm an example function"

    # Test
    assert func(**params) == returns
