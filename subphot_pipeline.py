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

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def first_image_hdu(hdul):
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
            return hdu
    raise ValueError("No 2-D image HDU found.")


def header_value(hdr, keys, default=None, cast=float):
    for k in keys:
        if k in hdr:
            try:
                return cast(hdr[k])
            except (TypeError, ValueError):
                return hdr[k]
    return default

# colors

_SLOAN_MAP = {"up": "u", "gp": "g", "rp": "r", "ip": "i", "zp": "z", "zs": "z",
              "u": "u", "g": "g", "r": "r", "i": "i", "z": "z"}
_JOHNSON = {"U", "B", "V", "R", "I"}

# system label used by GaiaXPy column prefixes
BAND_SYSTEM = {**{b: "SDSS" for b in "ugriz"},
               **{b: "JKC" for b in _JOHNSON}}
SYSTEM_BANDS = {"SDSS": list("ugriz"), "JKC": list(_JOHNSON)}
SYSTEM_LABEL = {"SDSS": "Sdss", "JKC": "Jkc"}

# colour pair (bluer, redder) used to derive each band's colour term
COLOR_PAIR = {"u": ("u", "g"), "g": ("g", "r"), "r": ("g", "r"),
              "i": ("r", "i"), "z": ("i", "z"),
              "U": ("U", "B"), "B": ("B", "V"), "V": ("B", "V"),
              "R": ("V", "R"), "I": ("R", "I")}


def normalize_band(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in _JOHNSON:                      # exact 'U','B','V','R','I'
        return s
    low = s.lower()
    for ch in ("'", '"', "-", "_", " "):
        low = low.replace(ch, "")
    low = low.replace("sdss", "").replace("pan-starrs", "").replace("ps1", "")
    if low in _SLOAN_MAP:
        return _SLOAN_MAP[low]
    su = s.upper()
    if any(t in su for t in ("BESS", "JOHNSON", "COUSINS")):
        for j in ("U", "B", "V", "R", "I"):
            if su.rstrip().endswith(j):
                return j
    return None


def peek_band(path: str):
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                for k in ("FILTER", "FILTER1", "FILTER2", "BAND"):
                    if k in hdu.header and str(hdu.header[k]).strip():
                        raw = str(hdu.header[k]).strip()
                        return normalize_band(raw), raw
    except Exception:
        pass
    return None, None

# Image loading / metadata
@dataclass
class Frame:
    path: str
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    exptime: float
    mjd: float
    gain: float
    rdnoise: float
    saturate: float
    band: str
    name: str = field(default="")


def load_frame(path: str, cfg: Config) -> Frame:
    with fits.open(path) as hdul:
        hdu = first_image_hdu(hdul)
        data = np.asarray(hdu.data, dtype=np.float64)
        hdr = hdu.header
    wcs = WCS(hdr)

    exptime = header_value(hdr, ["EXPTIME", "EXPOSURE"], 1.0)
    mjd = header_value(hdr, ["MJD-OBS", "MJD", "MJDOBS"], None)
    if mjd is None:
        dateobs = header_value(hdr, ["DATE-OBS", "DATE_OBS"], None, cast=str)
        mjd = Time(dateobs).mjd if dateobs else np.nan

    # BANZAI e91 frames are already in electrons -> effective gain ~ 1.
    gain = cfg.gain if cfg.gain is not None else header_value(
        hdr, ["GAIN", "EGAIN", "CCDGAIN"], 1.0)
    rdnoise = cfg.rdnoise if cfg.rdnoise is not None else header_value(
        hdr, ["RDNOISE", "RDNOIS", "READNOIS"], 10.0)
    saturate = cfg.saturate if cfg.saturate is not None else header_value(
        hdr, ["SATURATE", "MAXLIN", "SATLEVEL"], 60000.0)
    band = header_value(hdr, ["FILTER", "FILTER1", "BAND"], "g", cast=str)

    return Frame(path=path, data=data, header=hdr, wcs=wcs, exptime=exptime,
                 mjd=float(mjd), gain=float(gain), rdnoise=float(rdnoise),
                 saturate=float(saturate), band=str(band),
                 name=os.path.splitext(os.path.basename(path))[0])