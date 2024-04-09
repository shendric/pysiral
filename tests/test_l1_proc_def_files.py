# -*- coding: utf-8 -*-
"""
Testing all Level-1p product definition files for format compliance with the
applicable pydantic data model
"""

import unittest
from loguru import logger
from pysiral import psrlcfg
from pydantic import ValidationError

logger.disable("pysiral")


class TestL1ProcDef(unittest.TestCase):

    def setUp(self):
        self.l1_proc_ids = psrlcfg.procdef.l1.keys()

    def testDefinitionFileFormatCompliance(self):
        from pysiral.l1 import L1pProcessorConfig
        for l1_proc_id in self.l1_proc_ids:
            try:
                _ = L1pProcessorConfig.from_yaml(l1_proc_id)
                l1p_proc_def_file_validated = True
                error_text = ""
                print(f"{l1_proc_id=} -> ok")
            except ValidationError as errors:
                l1p_proc_def_file_validated = False
                error_text = "\n"+errors.json(indent=2)
            self.assertTrue(l1p_proc_def_file_validated, msg=f"{l1_proc_id=}{error_text}")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestL1ProcDef)
    unittest.TextTestRunner(verbosity=2).run(suite)
