# -*- coding: utf-8 -*-

"""
Module containing class for data set id definitions
"""
import parse
from pydantic import BaseModel, PrivateAttr, field_validator


# noinspection PyNestedDecorators
class SourceDataID(BaseModel):
    """
    Dataset ID for radar altimeter source data
    """
    platform_or_mission: str
    timeliness: str
    publisher: str
    version: str
    _parser_str: str = PrivateAttr(default="{platform_or_mission}_{timeliness}_{publisher}_{version}")

    @field_validator("platform_or_mission", "timeliness", "publisher", "version")
    @classmethod
    def is_instance(cls, str_value: str) -> str:
        assert str_value.isidentifier()
        return str_value

    @classmethod
    def from_str(cls, version_str: str) -> "SourceDataID":
        """
        Initialize the class from a version string

        :param version_str:

        :return:
        """
        parser_str = cls._parser_str.get_default()
        result = parse.parse(parser_str, version_str)
        if not result:
            raise ValueError(f"{cls.__class__.__name__}: {version_str=} not following convention {parser_str}")
        return cls(**result.named)

    @property
    def version_str(self) -> str:
        return self._parser_str.format(**dict(self))


# noinspection PyNestedDecorators
class L1PDataID(BaseModel):
    """
    Version for Level-1P dataset
    """
    platform: str
    timeliness: str
    source_dataset: str
    version: str
    _parser_str: str = PrivateAttr(default="{platform}_{timeliness}_{source_dataset}_{version}")

    @field_validator("platform", "timeliness", "source_dataset", "version")
    @classmethod
    def is_instance(cls, str_value: str) -> str:
        assert str_value.isidentifier()
        return str_value

    @classmethod
    def from_str(cls, version_str: str) -> "L1PDataID":
        """
        Initialize the class from a version string

        :param version_str:

        :return:
        """
        parser_str = cls._parser_str.get_default()
        result = parse.parse(parser_str, version_str)
        if not result:
            raise ValueError(f"{cls.__class__.__name__}: {version_str=} not following convention {parser_str}")
        return cls(**result.named)

    @property
    def version_str(self) -> str:
        return self._parser_str.format(**dict(self))
