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

#just to remove some warnings
warnings.simplefilter("ignore")

# right now this code is ment for LCO images

@dataclass
class Config:
    sci_glob: str                      #sciene images path
    ref_path: str                      # reference images path
    ra: float
    dec: float
    outdir: str = "out"                # output directory
    hotpants: str = "hotpants"         # path of hotpants executable file
    gain: float | None = None          # e-/ADU; None -> read from header
    rdnoise: float | None = None       # e-;     None -> read from header
    saturate: float | None = None      # ADU;    None -> read from header
    saturate_frac: float = 0.90        # use 0.9x sat as HOTPANTS upper limit
    kernel_order: int = 2              # HOTPANTS -ko spatial kernel order
    flatten_bkg: bool = True           # remove large-scale gradients (keep pedestal)
    bkg_box: int = 128                 # Background2D mesh size (px) -- keep LARGE
    bkg_filter: int = 3                # Background2D median filter size (meshes)
    bkg_order: int = 1                 # HOTPANTS -bgo spatial background order
    aper_fwhm: float = 1.5             # photometry aperture radius in FWHM
    big_aper_fwhm: float = 5.0         # curve-of-growth "total" aperture in FWHM
    sky_in_fwhm: float = 4.0           # sky annulus inner radius in FWHM
    sky_out_fwhm: float = 7.0          # sky annulus outer radius in FWHM
    cal_radius_arcmin: float = 0.0     # 0 -> auto from frame footprint
    cal_gmin: float = 13.0             # bright cut (Gaia G) for calibrators
    cal_gmax: float = 17.5             # faint cut (GSPC reliable to G~17.65)
    cal_max_stars: int = 200           # cap calibrators (fast, plenty for a ZP)
    cstar_max: float = 0.10            # GSPC |C*| quality cut (smaller=cleaner)
    use_gaiaxpy: bool = False          # True -> integrate spectra instead of GSPC; This takes so much time!!
    match_arcsec: float = 1.0          # science<->Gaia matching radius
    fit_color_term: bool = True
    sn_color: float | None = None      # SN colour (band's pair) for correction
    nsigma_limit: float = 5.0          # upper-limit confidence
    ncpu: int = 1                      # parallel workers over epochs
    cutouts: bool = True               # write sci/ref/diff triptych per epoch
    plots: bool = False                # plotting

