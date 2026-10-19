# -*- coding: utf-8 -*-


import argparse

from typing import List, Optional

from dateperiods import DatePeriod


from pysiral import PSRLCFG, set_psrl_cpu_count
from pysiral.scripts.parser_items import (
    ProcessingPeriod, ExcludeMonths, Hemisphere, PlatformID,
    ProductID, MultiProcesssingNumCores, UseMultiProcesssing
)
from pysiral.scripts.l1preproc import l1preproc


def l1preproc_product(
    product_id: str,
    processing_period: DatePeriod,
    platform: str,
    hemisphere: str = "global",
    exclude_months: Optional[List[int]] = None,
    use_multiprocessing: bool = False,
    multiprocessing_num_cores: Optional[int] = None
) -> None:
    """
    Workflow script of the pysiral l1b preprocessor.

    :param product_id: The product ID to process.
    :param processing_period: The processing period to run the preprocessor for.
    :param exclude_months: List of months to exclude from processing (optional).
    :param hemisphere: Hemisphere to process data for, default is "global".
    :param platform: Platform identifier (mandatory for product based processing).
    :param use_multiprocessing: Flag to enable multiprocessing (default is False).
    :param multiprocessing_num_cores: Number of cores to use for multiprocessing (optional).

    :return: None
    """

    # Get the Level-1 preprocessor settings for the given product ID and context variabes
    context = {
        "product_id": product_id,
        "platform": platform,
        "hemisphere": hemisphere,
    }
    l1p_settings = PSRLCFG.get_product_settings("l1", product_id, context)

    # Update settings with pass-through variables
    pass_through_keyw = {
        "processing_period": processing_period,
        "exclude_months": exclude_months,
        "use_multiprocessing": use_multiprocessing,
        "multiprocessing_num_cores": multiprocessing_num_cores,
    }
    l1p_settings.update(pass_through_keyw)

    # Run regular l1preproc with the settings obtained from the product definition
    l1preproc(**l1p_settings)

class L1PreProcProductScriptArguments(object):

    def __init__(self):
        """
        Command line parser class for the pysiral Level-1 Pre-Processor
        """
        self.parser = self.get_argument_parser()

    def get(self, args_list: Optional[List[str]] = None) -> "argparse.Namespace":
        args = self.parser.parse_args() if args_list is None else self.parser.parse_args(args_list)
        if args.multiprocessing_num_cores is not None:
            set_psrl_cpu_count(args.multiprocessing_num_cores)
        return args

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Set up the command line argument parser for the Level-1 Pre-Processor.

        :return: The argument parser object.
        """

        # Level-1 pre-processor specific help text for the hemisphere argument
        hemisphere_l1p_help = """
        Target hemisphere for processing. Options are 'global', 'nh', or 'sh'. The 
        latitude limit of the hemisphere is defined in the Level-1 pre-processor settings file.
        If 'global' is selected, the processor will run for both hemispheres, but still within the
        latitude limits.
        """

        # List of command line option required for the Level-1 pre-processor
        arg_item_list = [
            # Positional arguments
            ProductID(),
            ProcessingPeriod(),
            # Optional arguments
            ExcludeMonths(),
            Hemisphere(help=hemisphere_l1p_help),
            PlatformID(),
            UseMultiProcesssing(),
            MultiProcesssingNumCores()
        ]

        # create the parser
        parser = argparse.ArgumentParser(
            prog="pysiral l1",
            description="""
            The Level-1 Pre-Processor (l1) is used to generate Level-1 files (l1p) from 
            individual source radar altimeter files for a given period. Processing steps include the 
            harmonization of data formats from various radar altimeter missions, the generation of 
            continuous trajectories over the polar oceans, ingesting auxiliary data and the pre-computation 
            of waveform shape parameters. Basis for the Level-1 Pre-Processor is a Level-1 Processor 
            Definition file that defines the the radar altimeter input and processing parameters.
            The output is a set of Level-1 files (l1p) that contain the  harmonized data from the 
            source files, which can be used for further processing in the Level-2 processor (l2).
            """,
            epilog="For more information, see: https://pysiral.readthedocs.io",
            formatter_class=lambda prog: argparse.HelpFormatter(prog, width=96, indent_increment=4)  # noqa: E501
        )
        for arg_item in arg_item_list:
            arg_flags, args_dict = arg_item.get()
            parser.add_argument(*arg_flags, **args_dict)

        return parser
