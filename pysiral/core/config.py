# -*- coding: utf-8 -*-
#
# Copyright © 2015 Stefan Hendricks
#
# Licensed under the terms of the GNU GENERAL PUBLIC LICENSE
#
# (see LICENSE for details)

"""
Purpose:
    Returns content of configuration and definition files

Created on Mon Jul 06 10:38:41 2015

@author: Stefan
"""

from pathlib import Path
from typing import Dict, Union
from ruamel.yaml import YAML
from packaging.version import Version


class DataVersion(Version):
    """
    A small modification of packaging.version.Version
    allowing conversion to and from filenames
    (1.0 -> v1p0 -> 1.0)
    """

    def __init__(self, version_str: str) -> None:
        version_str = str(version_str).replace("v", "")
        version_str = version_str.replace("p", ".")
        super(DataVersion, self).__init__(version_str)

    @property
    def filename(self) -> str:
        file_version = self.public.replace(".", "p")
        return f"v{file_version}"


# TODO: Marked as obsolete -> flag_dict now in mission_def yaml.
class RadarModes(object):

    flag_dict = {"lrm": 0, "sar": 1, "sin": 2}

    def __init__(self):
        pass

    @classmethod
    def get_flag(cls, mode_name: str) -> Union[int, None]:
        try:
            return cls.flag_dict[mode_name]
        except KeyError:
            return None

    @classmethod
    def get_name(cls, flag: int) -> Union[str, None]:
        return next(
            (
                mode_name
                for mode_name, mode_flag in cls.flag_dict.items()
                if flag == mode_flag
            ),
            None,
        )

    def name(self, index: int) -> str:
        i = list(self.flag_dict.values()).index(index)
        return list(self.flag_dict.keys())[i]

    @property
    def num(self) -> int:
        return len(self.flag_dict.keys())


def get_yaml_as_dict(filepath: Path) -> Dict:
    """
    Parses the content of a yaml file into a dictionary

    :param filepath: File path to yaml file

    :return: Dictionary
    """
    reader = YAML(typ="safe", pure=True)  # YAML 1.2 support
    with filepath.open() as f:
        content_dict = reader.load(f)
    return content_dict
