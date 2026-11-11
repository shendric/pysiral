# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:57:56 2016

@author: Stefan

Important Note:

    All sitype data handlers must be subclasses of pysiral.auxdata.AuxdataBaseClass in order to work
    for the Level-2 Processor. If the auxiliary class is based on a static dataset, this should be parsed
    in `__init__`.

    Please review the variables and properties in the parent class, as well as the correspodning config and
    support classes for grid track interpolation in the pysiral.auxdata module for additional guidance.

    The only other hard requirements is the presence of on specific method in order to be a valid subclass of
    AuxdataBaseClass:


        get_l2_track_vars(l2)

            This method will be called during the Level-2 processor. The argument is the Level-2 data object and
            the purpose of the method is to compute the auxilary variable(s) and associated uncertainty. These
            variable need to be registered using the `register_auxvar(id, name, value, uncertainty)` method of
            the base class. All sitype subclasses need to register at minimum the following variable:

                sea ice type (fraction of multi year ice):
                    id: sitype
                    name: sea_ice_type

            e.g., this code line is mandatory for `get_l2_track_vars` (uncertainty can be None):

                # Register Variables
                self.register_auxvar("sitype", "sea_ice_type", value, uncertainty)

"""


import contextlib
import numpy as np

from pysiral.auxdata import AuxDataLoader, AuxDataExitFlag


class MYIDefault(AuxDataLoader):
    """ Returns myi for all ice covered regions """

    def __init__(self, cfg):
        self.cfg = cfg
        super(MYIDefault, self).__init__()

    def apply(self, l1, l2):
        """ Every ice is myi (sitype = 1) """
        sitype = np.zeros(shape=l2.sic.shape, dtype=np.float32)
        is_ice = np.where(l2.sic > 0)[0]
        sitype[is_ice] = 1.0
        uncertainty = np.full(sitype.shape, self.uncertainty_default)
        # Register the data
        l2.set_auxiliary_parameter("sitype", "sea_ice_type", sitype, uncertainty)

    @property
    def uncertainty_default(self):
        if "uncertainty_default" in self.cfg.options:
            return self.cfg.options.uncertainty_default
        else:
            return 0.0


class FYIDefault(AuxDataLoader):
    """
    Returns fyi for all ice covered regions
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        super(FYIDefault, self).__init__()

    def apply(self, l1, l2):
        """ Every ice is fyi (sitype = 0) """
        sitype = np.zeros(shape=l2.sic.shape, dtype=np.float32)
        uncertainty = np.full(sitype.shape, self.uncertainty_default)
        l2.set_auxiliary_parameter("sitype", "sea_ice_type", sitype, uncertainty)
        return AuxDataExitFlag.NOMINAL

    @property
    def uncertainty_default(self):
        with contextlib.suppress(TypeError):
            if "uncertainty_default" in self.cfg.options:
                return self.cfg.options.uncertainty_default
        return 0.0

