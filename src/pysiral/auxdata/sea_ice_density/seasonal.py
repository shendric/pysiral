# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["SeasonalSeaIceDensityClimatology"]


from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Dict, Any, List, Optional, Literal

import numpy as np
import pandas as pd

from pysiral.auxdata import AuxDataLoader, AuxDataExitFlag



class SeasonalSeaIceDensityConfig(BaseModel):
    """
    Configuration class for the SeasonalSeaIceDensityClimatology auxiliary data loader.
    """
    date_list: list[Annotated[str, Field(
        description="List of dates [%m-%d] corresponding to the seasonal sea ice density values.",
    )]]
    value_list: list[Annotated[float, Field(
        ...,
        description="List of seasonal sea ice density values in kg/m^3 corresponding to the dates.",
        gt=600.0, lt=1000.0,
    )]]
    interpolation_method: Literal["linear"] = Field(
        description="Temporal interpolation method",

    )
    sea_ice_density_uncertainty: float = Field(
        ...,
        description="The uncertainty associated with the sea ice density in kg/m^3.",
        gt=0.0, lt=600.0,
    )

    @classmethod
    @field_validator("date_list")
    def validate_date_list(cls, v):
        for date_str in v:
            try:
                datetime.strptime(date_str, "%d-%m")
            except ValueError:
                raise ValueError(f"Invalid date format: {date_str}. Expected format: %d-%m")
        return v


class SeasonalSeaIceDensityClimatology(AuxDataLoader):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(SeasonalSeaIceDensityClimatology, self).__init__()
        self.cfg = SeasonalSeaIceDensityConfig(**cfg.options)

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
        sidens = self.get_seasonal_sea_ice_density(l2)
        sidens_uncertainty = np.full(sidens.shape, self.cfg.sea_ice_density_uncertainty)

        # Register the auxiliary variable
        l2.set_auxiliary_parameter("idens", "sea_ice_density", sidens, sidens_uncertainty)
        return exit_status

    def get_seasonal_sea_ice_density(self, l2) -> np.ndarray:
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