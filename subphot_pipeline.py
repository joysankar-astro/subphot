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

def flatten_background(data: np.ndarray, box: int, filt: int,
                       fwhm: float = 4.0, preserve_pedestal: bool = True):
    """
    Remove only the LARGE-SCALE spatial structure of the sky (gradients,
    scattered light, moon, fringing residuals) while keeping the mean sky
    pedestal intact.
    """
    from photutils.background import Background2D, MedianBackground
    from scipy.ndimage import binary_dilation

    _, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    # mask stars + galaxy cores so they don't bias the background mesh
    src_mask = (data - med) > 3.0 * std
    src_mask = binary_dilation(src_mask, iterations=max(int(round(fwhm)), 2))

    box = int(min(box, data.shape[0] // 4, data.shape[1] // 4))
    box = max(box, 32)
    try:
        bkg = Background2D(
            data, box_size=box, filter_size=filt,
            sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground(),
            mask=src_mask, exclude_percentile=90.0)
        model = bkg.background
    except Exception as exc:
        log(f"Background2D failed ({exc}); skipping flattening.")
        return data, 0.0

    pedestal = float(np.median(model)) if preserve_pedestal else 0.0
    flattened = data - model + pedestal
    structure_rms = float(np.std(model - np.median(model)))
    return flattened, structure_rms



# fix columns
def _xy_cols(tbl):
    xc = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    yc = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    return xc, yc

def measure_fwhm_bkg(data: np.ndarray, fwhm_guess: float = 4.0):
    """Return (fwhm_pixels, sky_median, sky_std, star_table)."""
    from photutils.detection import DAOStarFinder
    from photutils.morphology import data_properties

    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    finder = DAOStarFinder(fwhm=fwhm_guess, threshold=8.0 * std,
                           exclude_border=True)
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return fwhm_guess, median, std, Table()

    sources.sort("flux")
    sources.reverse()
    bright = sources[: min(40, len(sources))]

    # robust FWHM from second moments of bright, isolated, unsaturated stars
    fwhms = []
    half = 9
    ny, nx = data.shape
    xc, yc = _xy_cols(bright)
    for row in bright:
        x, y = int(round(row[xc])), int(round(row[yc]))
        if x - half < 0 or x + half >= nx or y - half < 0 or y + half >= ny:
            continue
        cut = data[y - half:y + half + 1, x - half:x + half + 1] - median
        if cut.max() <= 0:
            continue
        try:
            props = data_properties(cut)
            sx = props.semimajor_sigma.value
            sy = props.semiminor_sigma.value
            fw = 2.3548 * np.sqrt(sx * sy)
            if 1.0 < fw < 25.0:
                fwhms.append(fw)
        except Exception:
            continue
    fwhm = float(np.nanmedian(fwhms)) if fwhms else fwhm_guess
    return fwhm, float(median), float(std), sources

NO_DATA = 0.0

def reproject_reference(ref: Frame, sci: Frame, out_path: str) -> str:
    array = footprint = None
    try:
        from reproject import reproject_adaptive
        array, footprint = reproject_adaptive(
            (ref.data, ref.wcs), sci.wcs, shape_out=sci.data.shape,
            roundtrip_coords=True)
    except Exception:
        array = None
    if array is None:
        from reproject import reproject_interp
        array, footprint = reproject_interp(
            (ref.data, ref.wcs), sci.wcs, shape_out=sci.data.shape)

    array = np.asarray(array, dtype=np.float32)
    if footprint is not None:
        array[~footprint.astype(bool)] = NO_DATA      # outside coverage -> 0
    array[~np.isfinite(array)] = NO_DATA              # NaN/Inf -> 0 (HOTPANTS-safe)

    hdr = sci.wcs.to_header()
    hdr["EXPTIME"] = ref.exptime
    hdr["GAIN"] = ref.gain
    hdr["RDNOISE"] = ref.rdnoise
    hdr["SATURATE"] = ref.saturate
    fits.writeto(out_path, array, hdr, overwrite=True)
    return out_path


# HOTPANTS subtraction
def write_plain_fits(frame: Frame, out_path: str) -> str:
    """Write science as a single-HDU plain FITS, NaN/Inf cleaned to 0.0.
    """
    data = np.asarray(frame.data, np.float32).copy()
    data[~np.isfinite(data)] = NO_DATA
    hdr = frame.wcs.to_header()
    hdr["EXPTIME"] = frame.exptime
    fits.writeto(out_path, data, hdr, overwrite=True)
    return out_path


def run_hotpants(sci_path: str, ref_path: str, out_diff: str, out_noise: str,
                 sci: Frame, ref: Frame, sci_fwhm: float, ref_fwhm: float,
                 cfg: Config) -> bool:
    
    # kernel half-width ~ 2.5 x broadest sigma; Gaussian basis scaled to seeing.
    broad_fwhm = max(sci_fwhm, ref_fwhm)
    sigma = broad_fwhm / 2.3548
    r_kernel = int(np.clip(round(2.5 * broad_fwhm), 8, 25))
    rss = int(r_kernel * 1.5)

    # match-up direction: blur the sharper frame
    convolve_template = ref_fwhm <= sci_fwhm  # ref sharper -> convolve template

    # valid-data lower limits: data still carries its sky pedestal (we only
    # flattened the structure), so set the floor below the sky distribution.
    _, sci_med, sci_std = sigma_clipped_stats(sci.data, sigma=3)
    _, ref_med, ref_std = sigma_clipped_stats(ref.data, sigma=3)
    sci_lo = sci_med - 5.0 * sci_std
    ref_lo = ref_med - 5.0 * ref_std

    # three-Gaussian basis scaled to the matching kernel (sigmas in pixels)
    s = max(sigma * 0.5, 0.7)
    ng = ["3", "6", f"{0.5*s:.3f}", "4", f"{1.0*s:.3f}", "2", f"{2.0*s:.3f}"]

    # upper good-data limit at saturate_frac x saturation (avoid non-linear
    # regime), following AutoPhOT's saturate_frac=0.90 default.
    iu = cfg.saturate_frac * sci.saturate
    tu = cfg.saturate_frac * ref.saturate

    cmd = [
        cfg.hotpants,
        "-inim", sci_path,
        "-tmplim", ref_path,
        "-outim", out_diff,
        "-oni", out_noise,                       # output noise image
        "-n", "i",                               # normalise to science
        "-c", "t" if convolve_template else "i",
        "-iu", f"{iu:.1f}", "-il", f"{sci_lo:.1f}",
        "-tu", f"{tu:.1f}", "-tl", f"{ref_lo:.1f}",
        "-ig", f"{sci.gain:.4f}", "-ir", f"{sci.rdnoise:.3f}",
        "-tg", f"{ref.gain:.4f}", "-tr", f"{ref.rdnoise:.3f}",
        "-r", str(r_kernel), "-rss", str(rss),
        "-ko", str(cfg.kernel_order),             # spatial kernel order
        "-bgo", str(cfg.bkg_order),               # spatial background order
        "-ng", *ng,
        "-v", "0",
    ]
    log("HOTPANTS: convolving %s (sci FWHM=%.2f px, ref FWHM=%.2f px)"
        % ("template" if convolve_template else "science", sci_fwhm, ref_fwhm))

    # Help the HOTPANTS binary find its shared libs (e.g. libcfitsio.so.10):
    # prepend the conda env lib dir and the binary's own dir to LD_LIBRARY_PATH.
    env = os.environ.copy()
    libdirs = []
    if env.get("CONDA_PREFIX"):
        libdirs.append(os.path.join(env["CONDA_PREFIX"], "lib"))
    bindir = os.path.dirname(os.path.abspath(cfg.hotpants)) if os.sep in cfg.hotpants \
        else ""
    if bindir:
        libdirs += [bindir, os.path.join(os.path.dirname(bindir), "lib")]
    if libdirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            libdirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))

    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=900, env=env)
    except FileNotFoundError:
        log("ERROR: HOTPANTS binary not found. Install it or pass --hotpants.")
        return False
    except subprocess.TimeoutExpired:
        log("ERROR: HOTPANTS timed out.")
        return False

    if not os.path.exists(out_diff):
        msg = (res.stderr or res.stdout or "")[-800:]
        log("HOTPANTS failed:\n" + msg)
        if "shared libraries" in msg or "cannot open shared object" in msg:
            log("HINT: a shared library (e.g. libcfitsio) is not on the loader "
                "path. Try:  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
        return False
    return convolve_template
