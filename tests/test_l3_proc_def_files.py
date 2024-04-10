# -*- coding: utf-8 -*-
"""
Testing all Level-2 processor definition files for compabilility with
Level2Processor conventions

@author: Stefan Hendricks
"""

import unittest

from loguru import logger

from pysiral import psrlcfg
from pysiral.core.config import get_yaml_as_dict

logger.disable("pysiral")


class TestL3ProcDef(unittest.TestCase):

    def testYamlSyntaxOfDefinitionFiles(self):
        for procdef_id, procdef_entry in psrlcfg.procdef.l3.items():
            content = get_yaml_as_dict(procdef_entry.filepath)
            self.assertIsInstance(content, dict, msg=procdef_entry.filepath)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL3ProcDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
