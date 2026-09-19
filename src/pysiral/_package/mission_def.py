# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import yaml
from datetime import datetime
from loguru import logger
from typing import Union, Dict

try:
    from datetime import UTC
except ImportError:
    import timezone
    UTC = timezone.utc



class _MissionDefinitionCatalogue(object):
    """
    Container for storing and querying information from mission_def.yaml
    """

    def __init__(self, filepath):
        """
        Create a catalogue for all altimeter missions definitions
        :param filepath:
        """

        # Store Argument
        self._filepath = filepath

        # Read the file and store the content
        self._content = None
        with open(str(self._filepath)) as fh:
            self._content = yaml.safe_load(fh)

    def get_platform_info(self, platform_id) -> Union[Dict, None]:
        """
        Return the full configuration attr dict for a given platform id

        :param platform_id:

        :return:
        """
        platform_info = self._content.platforms.get(platform_id, None)
        return platform_info if platform_info is None else platform_info

    def get_platform_id(self, platform_name: str) -> Union[str, None]:
        """
        Return the name of a platform.
        :param platform_name:
        :return:
        """

        # Query the source dictionary
        platforms = [entry for entry in self._content.platforms.items() if entry[1]["long_name"] == platform_name]

        # No valid entry found -> Warning and returning None
        if not platforms:
            logger.warning(f"Did not find entry for {platform_name} in {self._filepath}")
            return None

        # Multiple Entries -> Error in configuration: Raise Exception
        elif len(platforms) > 1:
            msg = f"Multitple entries found for {platform_name} in {self._filepath}"
            logger.error(msg)
            raise ValueError(msg)

        platform_id, _ = platforms[0]
        return platform_id

    def get_name(self, platform_id):
        """
        Return the name of a platform.
        :param platform_id:
        :return:
        """
        platform_info = self.get_platform_info(platform_id)
        return None if platform_info is None else platform_info.long_name

    def get_sensor(self, platform_id):
        """
        Return the sensor name of a platform
        :param platform_id:
        :return:
        """
        platform_info = self.get_platform_info(platform_id)
        return None if platform_info is None else platform_info.sensor

    def get_orbit_inclination(self, platform_id):
        """
        Return the orbit inclination of a platform
        :param platform_id:
        :return:
        """
        platform_info = self.get_platform_info(platform_id)
        return None if platform_info is None else platform_info.orbit_max_latitude

    def get_time_coverage(self, platform_id):
        """
        Get the time coverage (start and end of data coverage) of the requested plaform.
        If the end data is not defined because the platform is still active, the current
        date is returned.
        :param platform_id:
        :return: time coverage start & time coverage end
        """
        platform_info = self.get_platform_info(platform_id)
        if platform_info is None:
            return None, None
        tcs = platform_info.time_coverage.start
        tce = platform_info.time_coverage.end
        if tce is None:
            tce = datetime.now(UTC)
        return tcs, tce

    @property
    def content(self) -> Dict:
        """
        The content of the definition file as an attribute-enabled dictionary.
        :return:
        """
        return self._content

    @property
    def ids(self):
        """
        A list of id's for each platform.

        :return: list with platform ids
        """
        return list(self.content.platforms.keys())
