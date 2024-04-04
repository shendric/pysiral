# -*- coding: utf-8 -*-
"""
Testing all Level-3 output definition files for compabilility with
Level2Processor conventions

#TODO: Add CF/ACDD compliance checks

@author: Stefan Hendricks
"""

import unittest

from loguru import logger

from pysiral import psrlcfg
from pysiral.core.config import get_yaml_as_dict

logger.disable("pysiral")


class TestL3OutputDef(unittest.TestCase):

    def setUp(self):

        # Get a list of processor definition files in the code repository
        self.l3_output_files = psrlcfg.get_settings_files("output", "l3")

    def testYamlSyntaxOfDefinitionFiles(self):
        for filename in self.l3_output_files:
            content = get_yaml_as_dict(filename)
            self.assertIsInstance(content, dict, msg=filename)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL3OutputDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
