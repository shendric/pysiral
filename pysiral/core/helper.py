# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:04:43 2015

@author: Stefan
TODO: Is this still being used?
"""

from typing import List, Tuple

import numpy as np
from dateutil import parser as dtparser


def get_multiprocessing_1d_array_chunks(
        array_size: int,
        n_processes: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Break down an array in chunks for multiprocessing. The rule
    is one chunk per process/CPU.

    :param array_size: The number of elements to be distributed to chunks
    :param n_processes: The number processes/CPU's/chunks

    :return: List with start and end index for each chunk and number
        of (viable) processes
    """

    # Can't have more processes than chunks
    n_processes = min(n_processes, array_size)

    # Compute the chunks
    chunk_size = int(np.ceil(float(array_size) / float(n_processes)))
    chunks = np.arange(0, array_size, chunk_size, dtype=int)

    # Chunks with only one element are special case
    if chunk_size == 1:
        return zip(chunks, chunks), n_processes

    # For small arrays, the number of chunks may vary from desired number
    # of processes (integer granularity) -> recompute `n_processes`
    n_processes = len(chunks)

    # Last chunk may be smaller
    if chunks[-1] != (array_size - 1):
        chunks = np.append(chunks, [array_size])
    else:
        chunks[-1] += 1

    return zip(chunks[:-1], chunks[1:] - 1), n_processes


def parse_datetime_str(dtstr):
    """ Converts a time string to a datetime object using dateutils """
    return dtparser.parse(dtstr)


def get_first_array_index(array, value):
    """ Get the index in array of the first occurance of ``value`` """
    try:
        index = list(array).index(value)
    except ValueError:
        index = None
    return index


def get_last_array_index(array, value):
    """ Get the index in array of the last occurance of ``value`` """
    listarray = list(array)
    try:
        index = (len(listarray) - 1) - listarray[::-1].index(value)
    except ValueError:
        index = None
    return index


def rle(inarray):
    """
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)

    from: http://stackoverflow.com/questions/1066758/find-length-of-sequences-
                 of-identical-values-in-a-numpy-array
    """
    ia = np.array(inarray)                   # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    y = np.array(ia[1:] != ia[:-1])      # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)    # must include last element posi
    z = np.diff(np.append(-1, i))        # run lengths
    p = np.cumsum(np.append(0, z))[:-1]  # positions
    return z, p, ia[i]


class ProgressIndicator(object):

    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.index = None
        self.reset()

    def reset(self):
        self.index = 0

    def get_status_report(self, i, fmt="{step} of {n_steps} ({percent:.2f}%)"):
        self.index = i
        return fmt.format(step=self.step, n_steps=self.n_steps, percent=self.percent)

    @property
    def step(self):
        return self.index+1

    @property
    def percent(self):
        return float(self.step)/float(self.n_steps)*100.
