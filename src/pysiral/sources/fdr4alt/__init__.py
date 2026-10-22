# -*- coding: utf-8 -*-

"""
This module contains data loaders and file discovery classes for the Envisat mission.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["data_loader", "file_discovery", "EnvisatFDR4ALTDataLoader", "ERSFDR4ALTDataLoader"]

from pysiral.sources.fdr4alt.data_loader import EnvisatFDR4ALTDataLoader, ERSFDR4ALTDataLoader
