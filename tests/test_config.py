# -*- coding: utf-8 -*-
"""
Testing the pysiral configuration management

@author: Stefan
"""

import unittest
from pathlib import Path

from loguru import logger

from pysiral import psrlcfg

logger.disable("pysiral")


class TestConfig(unittest.TestCase):

    def setUp(self):
        pass

    # def testMissionConfig(self):
    #     self.assertIsInstance(psrlcfg.platforms.content, AttrDict)
    #     self.assertIsInstance(psrlcfg.platforms.ids, list)

    def testConfigPath(self):
        self.assertTrue(Path(psrlcfg.package.config_path).is_dir())
        self.assertTrue(Path(psrlcfg.package.package_root).is_dir())
        self.assertTrue(Path(psrlcfg.package.user_home).is_dir())

    def testLocalMachineDefinition(self):
        """
        Test the local machine definition
        :return:
        """
        from pysiral._package_config._local_machine import LocalMachineConfig
        if psrlcfg.local_path.filepath is not None:
            self.assertTrue(psrlcfg.local_path.filepath.is_file())
            self.assertTrue(hasattr(psrlcfg.local_path, "model_config"))

    def testL1ProcessorDefinitions(self):
        """
        Test the processor definitions for L1 Pre-processor
        :return:
        """

        # Loop over all processor levels

        # for processor_level in psrlcfg.processor_levels:
        #
        #     # Get a list of all ids
        #     proc_defs = psrlcfg.get_processor_definition_ids(processor_level)
        #     label = f"procdef:{processor_level}"
        #
        #     # lists of ids must be a list and longer than 0
        #     self.assertIsInstance(
        #         proc_defs,
        #         list,
        #         msg=f"Type is not list: {type(proc_defs)} [{label}]",
        #     )
        #     self.assertGreater(len(proc_defs), 0, msg=f"No definitions found for {label}")
        #
        #     # Load all processor definitions (must return a valid AttrDict)
        #     for proc_def_id in proc_defs:
        #         filepath = psrlcfg.get_settings_file("proc", processor_level, proc_def_id)
        #         self.assertIsInstance(filepath, Path)
        #         self.assertTrue(filepath.is_file())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    unittest.TextTestRunner(verbosity=2).run(suite)
