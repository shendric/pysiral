

import numpy as np
from pysiral.auxdata import AuxDataLoader, AuxDataExitFlag


class FixedSnowDepthDensity(AuxDataLoader):
    """
    Returns constant depth & density (values from l2 processor definition).

    Example entry in l2 proc config files:

        - snow:
            name: constant
            options:
                snow_density_uncertainty: 50

    TODO: Add uncertainties
    """

    def __init__(self, cfg):
        super(FixedSnowDepthDensity, self).__init__()
        self.cfg = cfg

    def apply(self, l1, l2):
        snow_depth = np.full(l2.n_records, self.cfg.options.fixed_snow_depth)
        snow_density = np.full(l2.n_records, self.cfg.options.fixed_snow_density)
        l2.set_auxiliary_parameter("sd", "snow_depth", snow_depth, 0.0)
        l2.set_auxiliary_parameter("sdens", "snow_density", snow_density, 0.0)
        return AuxDataExitFlag.NOMINAL

