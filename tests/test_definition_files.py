# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 17:57:33 2015

@author: Stefan
"""

import datetime
import unittest

from loguru import logger

from pysiral import psrlcfg
from pysiral.core.config import get_yaml_as_dict

logger.disable("pysiral")

# TODO: Find and add syntax test for all yaml files


class TestDefinitionfiles(unittest.TestCase):

    def setUp(self):
        pass

    def testYamlSyntaxOfAuxDataDefinitionFile(self):
        filename = psrlcfg.package.config_path / "auxdata_def.yaml"
        content = get_yaml_as_dict(filename)
        self.assertIsInstance(content, dict, msg=filename)

    def testMissionDefinitions(self):
        mission_ids = psrlcfg.missions.get_mission_ids()
        self.assertIsInstance(mission_ids, list)
        self.assertGreater(len(mission_ids), 0)

    def testPlatformDefinitions(self):
        platform_ids = psrlcfg.missions.get_platform_ids()
        self.assertIsInstance(platform_ids, list)
        self.assertGreater(len(platform_ids), 0)

    def testAuxdataDefinitionBasic(self):
        self.assertTrue(hasattr(psrlcfg, "auxdata"))
        keys = psrlcfg.auxdata.get_entries()
        self.assertGreater(len(keys), 0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDefinitionfiles)
    unittest.TextTestRunner(verbosity=2).run(suite)
