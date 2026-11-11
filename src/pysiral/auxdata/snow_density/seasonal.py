

from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from pysiral.auxdata import AuxdataBaseClass, AuxDataLoader, AuxDataExitFlag
from pysiral.auxdata.snow_depth import SnowParameterContainer


class SeasonalArcticSnowDensityMallett2020(AuxDataLoader):
    """
    Returns a seasonal varying but regionally static snow density from Mallett et al. 2020. The density
    is computed as a linear function of time (elapsed month since the October for the Arctic winter season):

        rho_snow = 6.50 * t + 274.51

    with t: month since October.

    Reference:

        Mallett, R. D. C., Lawrence, I. R., Stroeve, J. C., Landy, J. C., and Tsamados, M.:
        Brief communication: Conventional assumptions involving the speed of radar waves in snow
        introduce systematic underestimates to sea ice thickness and seasonal growth rate estimates,
        The Cryosphere, 14, 251–260, https://doi.org/10.5194/tc-14-251-2020, 2020.

    Notes:

        - This parametrization is valid stricly only for the northern hemisphere

        - Here we interpret the time t not as month since October but as fractional month
          (computed via days since Oct 15) to avoid artifical jumps of geophysical paramters
          between the last day of a month and the first of a next one.

    Example entry in l2 proc config files:

        - snow:
            name: snow_density_seasonal_mallett
            options:
                snow_density_uncertainty: 50.

    """

    def __init__(self, cfg):
        super(SeasonalArcticSnowDensityMallett2020, self).__init__()
        self.cfg = cfg

    def apply(self, l1, l2) -> AuxDataExitFlag:
        """
        Mandatory method for the Level-2 processor. Registers snow density (and only density)
        as the auxiliary parameter.
        :param l2:
        :return: AuxDataExitFlag

        """

        exit_status = AuxDataExitFlag.NOMINAL

        # Set the requested date
        self.set_requested_date_from_l2(l2)

        # --- Compute the factor t for the date of the l2 object ---
        date_tuple = self._requested_date
        winter_id = date_tuple[0] - int(date_tuple[1] < 10)   # The year of the winter in October

        # Number of days since October 15.
        epoch, l2date = datetime(winter_id, 10, 15), datetime(*date_tuple)
        n_days = (l2date-epoch).days

        # Compute the daily t-factor based on that t should the value of 6 on April 15.
        # of the winter seasons. There are 182 days between Oct. 15 and the following
        # 15th of April.
        #
        # Notes:
        #   - In case of a leap year the factor t will be slightly above 6 on April 15.
        #     This is intended.
        #   - The t factor will be negative before Oct. 15. This is intended
        t = float(n_days)/182. * 6.

        # Compute the fixed snow depth
        static_sdens = 6.5 * t + 274.51

        # Get the snow density uncertainty
        if "snow_density_uncertainty" in self.cfg.options:
            static_sdens_unc = self.cfg.options.snow_density_uncertainty
        else:
            exit_status = AuxDataExitFlag.ERROR_CONTINUE
            msg = ": Warning - snow_density_uncertainty missing in processor definition (set to 0.0)"
            logger.error(self.__class__.__name__ + msg)
            static_sdens_unc = 0.0

        # Create variables
        sdens = np.full(l2.n_records, static_sdens)
        sdens_unc = np.full(l2.n_records, static_sdens_unc)

        # Only use values for records, that are sea ice
        no_sea_ice = np.isnan(l2.sic[:])
        for var in [sdens, sdens_unc]:
            var[no_sea_ice] = np.nan

        # Register the auxiliary variable
        l2.set_auxiliary_parameter("sdens", "snow_density", sdens, sdens_unc)
        return exit_status


class SeasonalSnowDensityConfig(BaseModel):
    """
    Configuration class for the SeasonalSnowDensity auxiliary data loader.

    Attributes:
        snow_density_uncertainty (float): The uncertainty associated with the snow density.
    """
    date_list: list[Annotated[str, Field(
        description="List of dates [%m-%d] corresponding to the seasonal snow density values.",
    )]]
    value_list: list[Annotated[float, Field(
        ...,
        description="List of seasonal snow density values in kg/m^3 corresponding to the dates.",
        gt=0.0, lt=600.0,
    )]]
    snow_density_uncertainty: float = Field(
        ...,
        description="The uncertainty associated with the snow density in kg/m^3.",
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


class SeasonalSnowDensity(AuxDataLoader):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(SeasonalSnowDensity, self).__init__()
        self.cfg = SeasonalSnowDensityConfig(**cfg)

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