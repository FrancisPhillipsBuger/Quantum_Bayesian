#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Console reporting helpers: tag-prefixed log lines and throttled progress bars.

Created on: Sat Sep 19 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from tqdm import tqdm

#%% 1. Log lines
TAG_WIDTH = 13


def log(tag, msg):

    """

    Print one tag-prefixed line, e.g. "[Selection]  Best subset: ...".

    :param1 tag: stage label.
    :param2 msg: message text.

    :return: None.

    """

    print(f"{'[' + tag + ']':<{TAG_WIDTH}}{msg}", flush=True)


#%% 2. Progress bars
def progress(iterable, desc, total=None):

    """

    Wrap an iterable in a tqdm bar that refreshes at most every 30 s.

    :param1 iterable: items to iterate.
    :param2 desc:     short bar label.
    :param3 total:    item count when iterable has no len().

    :return: tqdm iterator.

    """

    return tqdm(iterable, desc=f"{' ' * TAG_WIDTH}{desc}", total=total,
                mininterval=30.0, maxinterval=60.0, ascii=True, ncols=100)
