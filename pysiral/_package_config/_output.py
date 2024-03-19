# -*- coding: utf-8 -*-

"""
Configuration data model for the output definition files
"""


from typing import Dict
from pydantic import BaseModel
from ._models import _YamlDefEntry


class OutputDefCatalog(BaseModel):
    l2i: Dict[str, _YamlDefEntry]
    l2p: Dict[str, _YamlDefEntry]
    l3: Dict[str, _YamlDefEntry]
