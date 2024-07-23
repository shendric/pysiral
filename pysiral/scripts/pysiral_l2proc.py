# -*- coding: utf-8 -*-

"""

"""

import argparse
from pysiral import psrlcfg
from typing import List, Callable, Literal
from dateperiods import DateDefinition, DatePeriod


def datedef_type(tcs_or_tce: Literal["tcs", "tce"]) -> Callable:
    """
    Factory function that provides additional option to the argparse directory type validation.

    :param tcs_or_tce: Defines type of DateDefinition

    :return: Function to validiates input for argparse
    """

    def datedef_type_func(int_list: List[int]) -> DateDefinition:
        """
        Small helper function to convert input automatically to
        Date Definition. Raises an exception if the input is invalid.

        :param int_list: Input argument (Expected: [yyyy[, mm][, [dd]])

        :raises argparse.ArgumentTypeError: raised if invalid input

        :return: Date Definition
        """
        try:
            date_def = DateDefinition(int_list, tcs_or_tce)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{int_list} is not a valid date definition`")
        return date_def

    return datedef_type_func


def dateperiod_type(int_list: List[int]) -> DatePeriod:
    """
    Small helper function to convert input automatically to
    Date Periood Raises an exception if the input is invalid.

    :param int_list: Input argument (Expected: [yyyy[, mm][, [dd]])

    :raises argparse.ArgumentTypeError: raised if invalid input

    :return: Date Definition
    """
    try:
        date_period = DatePeriod(int_list, int_list)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{int_list} is not a valid date definition`"
        ) from e
    return date_period


time_period_parser = argparse.ArgumentParser(add_help=False)
time_coverage_subparsers = time_period_parser.add_subparsers(
    title="Define time coverage",
    help="""
    The time coverage is either defined by a period (single day/month) 
    or by a start of end day/month.
    """,
    dest="time_coverage_type",
    required=True,
    metavar="{period | bounds}"
)
# create the parser for the "command_1" command
period_parser = time_coverage_subparsers.add_parser(
    'period',
    help="""
    Defines the time coverage by a single period.
    (`pysiral-l2proc period --help` for documentation)
    """
)
period_parser.add_argument(
    'period_definition',
    type=dateperiod_type,
    help=''
)

# create the parser for the "command_2" command
start_end_parser = time_coverage_subparsers.add_parser(
    "bounds",
    help="""
    Time coverage defined by start and end date.
    (`pysiral-l2proc bounds --help` for documentation) 
    """
)
start_end_parser.add_argument(
    "-s", "--start",
    type=datedef_type("tcs"),
    help='time coverage start'
)
start_end_parser.add_argument(
    "-e", "--end",
    type=datedef_type("tce"),
    action='store',
    help='test')

l2_proc_parser = argparse.ArgumentParser(
    prog="pysiral-l2proc",
    description="""
    Generates Level-2 (L2) products (Geophysical variables at full sensor resolution) from
    pre-processed Level-1 (L1P) data files. Requires the L2 and L1P dataset identifiers and 
    the target time coverage. 
    """,
    epilog="""
    Further information: https://pysiral.readthedocs.io/.
    """,
    allow_abbrev=False,
    formatter_class=argparse.RawTextHelpFormatter,
    parents=[time_period_parser]
)

l2_proc_parser.add_argument(
    "l2_dataset_id",
    metavar="l2_dataset_id",
    choices=psrlcfg.procdef.get_ids("l2"),
    help="""
    Identifier of the Level-2 data product. Defines the processor definition
    and output of the pysiral Level-2 processor. 
    (See https://pysiral.readthedocs.io/en/latest/core_concepts.html#dataset-id-s)
    """
)

l2_proc_parser.add_argument(
    "l1p_dataset_id",
    metavar="l1p_dataset_id",
    choices=[],
    help="""
    Identifier of the pre-processed Level-1 input data. 
    (See https://pysiral.readthedocs.io/en/latest/core_concepts.html#dataset-id-s)
    """
)

l2_proc_parser.add_argument(
    "-o", "--l2_output_ids",
    metavar="l2_output_ids",
    choices=[],
    required=False,
    help="""
    [Optional] Comma separated list of Level-2 output id's. Specifying this keyword
    allows to write multiple output files per Level-2 data object. It this keyword
    is not specified, only the default output will be written. If this keyword is
    specified and the default output is required, it needs to be included in the 
    list of output identifiers.   
    (See https://pysiral.readthedocs.io/en/latest/core_concepts.html#dataset-id-s)
    """
)


argps = l2_proc_parser.parse_args()
