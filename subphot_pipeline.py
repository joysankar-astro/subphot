from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import SigmaClip, sigma_clipped_stats, sigma_clip
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.wcs import WCS