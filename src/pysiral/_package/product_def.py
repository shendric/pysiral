# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from typing import Union, List, Dict
from pydantic import BaseModel



class ProductAttributes(BaseModel):
    """
    Data class to manage a product definition
    """
    product_line: str
    record_type: Union[str, List[str]]
    version: str


class ProductElements(BaseModel):
    """
    Data class to manage a product definition
    """
    platform: List[str]
    dynamic_variables: Dict[str, str]


class ProcessorDefinition(BaseModel):
    l1: Dict[str, Union[str, List[str]]]
    l2: Dict[str, Union[str, List[str]]]
    l2p: Dict[str, Union[str, List[str]]]
    l3: Dict[str, Union[str, List[str]]]


class ProductDefinition(BaseModel):
    attributes: ProductAttributes
    static_variables: dict
    product_elements: ProductElements
    processors: ProcessorDefinition



class ProductDefinitionCatalogue:
    """
    Container for the content of the product_def.yaml definition file
    for products
    """

    def __init__(self, lookup_directory) -> None:
        """
        Data class to manage a product definition catalogue
        :param lookup_directory: Path to the directory containing the product_def.yaml file
        """

        # Arguments
        self._lookup_directory = lookup_directory