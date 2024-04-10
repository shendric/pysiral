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


class TestL2OutputDef(unittest.TestCase):

    def testYamlSyntaxOfDefinitionFilesL2i(self):
        for procdef_id, procdef_entry in psrlcfg.outputdef.l2i.items():
            content = get_yaml_as_dict(procdef_entry.filepath)
            self.assertIsInstance(content, dict, msg=procdef_entry.filepath)

    def testYamlSyntaxOfDefinitionFilesL2p(self):
        for procdef_id, procdef_entry in psrlcfg.outputdef.l2p.items():
            content = get_yaml_as_dict(procdef_entry.filepath)
            self.assertIsInstance(content, dict, msg=procdef_entry.filepath)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL2OutputDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
