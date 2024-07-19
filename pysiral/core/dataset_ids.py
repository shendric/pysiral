# -*- coding: utf-8 -*-

"""
Module containing class for data set id definitions
"""
import parse
from typing import Literal
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


class L2ProcessingLevel(BaseModel):
    """
    Dataset identifier of Level-2 data products
    """
    processing_level: Literal["l2", "l2i"]
    product_line: str
    platform: str
    hemisphere: str
    timeliness: str
    version: str
    _parser_str: str = PrivateAttr(
        default="{processing_level}_{product_line}_{platform}_{hemisphere}_{timeliness}_{version}"
    )

    @field_validator(
        "processing_level", "product_line", "platform", "hemisphere", "timeliness", "version"
    )
    @classmethod
    def is_instance(cls, str_value: str) -> str:
        assert str_value.isidentifier()
        return str_value

    @classmethod
    def from_str(cls, version_str: str) -> "L2ProcessingLevel":
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
