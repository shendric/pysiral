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


class TestL2ProcDef(unittest.TestCase):

    def setUp(self):

        # Get a list of processor definition files in the code repository
        l2proc_ids = psrlcfg.get_setting_ids("proc", "l2")
        self.l2procdef_files = [psrlcfg.get_settings_file("proc", "l2", l2proc_id) for l2proc_id in l2proc_ids]

    def testYamlSyntaxOfDefinitionFiles(self):
        for filename in self.l2procdef_files:
            content = get_yaml_as_dict(filename)
            self.assertIsInstance(content, dict, msg=filename)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL2ProcDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
