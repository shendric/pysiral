# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from loguru import logger


class OptionalImportError(ImportError):
    """
    Exception for optional imports
    """
    def __init__(self, msg):
        logger.warning(msg)