from typing import Any

import numpy as np
from numpy.typing import NDArray

from engibench.problems.beams2d import Beams2D


def run_job(config: dict[str, Any]) -> NDArray[np.float64]:
    p = Beams2D()
    design, _ = p.optimize(config=config)
    return design
