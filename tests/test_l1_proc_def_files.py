# -*- coding: utf-8 -*-
"""
Testing all Level-2 output definition files for compabilility with
Level2Processor conventions

#TODO: Add CF/ACDD compliance checks

@author: Stefan Hendricks
"""

import unittest

from loguru import logger

from pysiral import psrlcfg
from pysiral.core.config import get_yaml_as_dict

logger.disable("pysiral")


class TestL1ProcDef(unittest.TestCase):

    def setUp(self):

        # Get a list of processor definition files in the code repository
        self.l1_proc_files = psrlcfg.get_settings_files("output", "l1")

    def testYamlSyntaxOfDefinitionFiles(self):
        for filename in self.l1_proc_files:
            content = get_yaml_as_dict(filename)
            self.assertIsInstance(content, dict, msg=filename)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL1ProcDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
