# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from pysiral.auxdata import AuxdataBaseClass, AuxDataLoader, AuxDataExitFlag
from pysiral.auxdata.snow_depth import SnowParameterContainer



class SeasonalSeaIceDensity(AuxDataLoader):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(SeasonalSeaIceDensity, self).__init__()
        self.cfg = SeasonalSeaIceDensityConfig(**cfg)

    def apply(self, l1, l2) -> AuxDataExitFlag:
        """
        Mandatory method for the Level-2 processor. Registers snow density (and only density)
        as the auxiliary parameter.

        :param l1: Level-1 data object
        :param l2: Level-2 data object

        :return: AuxDataExitFlag
        """
        exit_status = AuxDataExitFlag.NOMINAL

        # Get the seasonal snow density
        sdens = self.get_seasonal_snow_density(l2)
        sden_uncertainty = np.full(sdens.shape, self.cfg.snow_density_uncertainty)

        # Register the auxiliary variable
        l2.set_auxiliary_parameter("sdens", "snow_density", sdens, sden_uncertainty)
        return exit_status

    def get_seasonal_snow_density(self, l2) -> np.ndarray:
        """
        Computes the seasonal snow density based on the provided date_list and value_list in the configuration.

        :param l2: Level-2 data object

        :return: np.ndarray of seasonal snow density values corresponding to the dates in l2
        """
        l2_dates = pd.to_datetime(l2.time)
        start_year = l2_dates.min().year - 1
        end_year = l2_dates.max().year + 1
        season_dates = []
        for year in range(start_year, end_year + 1):
            for season in self.cfg.date_list:
                season_date = pd.to_datetime(f"{year}-{season}", format="%Y-%d-%m")
                season_dates.append(season_date)

        season_series = pd.Series(self.cfg.value_list * (end_year - start_year + 1), index=season_dates)
        season_series = season_series[~season_series.index.duplicated(keep='first')].sort_index()
        sdens = np.interp(l2_dates, season_series.index, season_series.values)
        return sdens