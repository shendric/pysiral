# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from pathlib import Path
from typing import Union, List, Dict, Optional

import yaml
from pydantic import BaseModel

from pysiral._package._models import ConvenientRootModel


class ProductConfig(BaseModel):
    """
    Data class to manage a product definition
    """
    pysiral_version: str


class ProductAttributes(BaseModel):
    """
    Data class to manage a product definition
    """
    product_line: str
    record_type: Union[str, List[str]]
    version: str
    platforms: List[str]


class ProductDynamicVariables(ConvenientRootModel):
    """
    Dynamic variables for a product definition
    """
    root: Dict[str, Dict[str, str]]

class ProcessorKeywords(ConvenientRootModel):
    root: Dict[str, Union[str, List[str]]]


class ProcessorDefinition(BaseModel):
    """
    Processor keywords for the fixed list of pysiral-supported processor types (l1, l2, l2p, l3)
    """
    l1: ProcessorKeywords
    l2: ProcessorKeywords
    l2p: ProcessorKeywords
    l3: ProcessorKeywords


class ProductDefinition(BaseModel):
    """
    Data model for the content of a product definition file (product_def.yaml)
    """
    config: ProductConfig
    attributes: ProductAttributes
    dynamic_variables: ProductDynamicVariables
    processors: ProcessorDefinition
    yaml_filepath: Optional[Union[str, Path]] = None


class ProductDefinitionCatalogue:
    """
    Container for the content of the product_def.yaml definition file
    for products
    """

    def __init__(self, lookup_directory: Path) -> None:
        """
        Data class to manage a product definition catalogue
        :param lookup_directory: Path to the directory containing the product_def.yaml file
        """

        # Arguments
        self._lookup_directory = lookup_directory
        self._product_definitions = self._get_product_definitions()

    def _get_product_definitions(self) -> Dict[str, ProductDefinition]:
        """
        Read the product_def.yaml file and return a dictionary with the product definitions
        :return: Dictionary with the product definitions
        """
        product_definition_files = list(sorted(Path(self._lookup_directory).glob("*.yaml")))

        # Create a dictionary with the product definitions
        product_definitions = {}
        for product_definition in product_definition_files:
            product_id = product_definition.stem
            with open(product_definition, "r") as fh:
                product_definitions[product_id] = ProductDefinition(**yaml.safe_load(fh))

        return product_definitions
