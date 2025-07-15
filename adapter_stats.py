import torch

import argparse
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from loguru import logger

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib

DEFAULT_N_SEEDS = 1