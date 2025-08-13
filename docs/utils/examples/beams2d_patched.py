import unittest.mock

import numpy as np

from examples.beams2d import run_job as real_run_job


def run_job(config):
    """To save resources, dont really optimize, just return zeros."""
    with unittest.mock.patch(
        "examples.beams2d.Beams2D.optimize", lambda obj, **_kwargs: (np.full((obj.nelx, obj.nely), 0.0), None)
    ):
        return real_run_job(config)
