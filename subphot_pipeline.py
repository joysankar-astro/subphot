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


def build_epsf(data: np.ndarray, sky: float, fwhm: float, satur: float):
    from photutils.detection import DAOStarFinder
    from photutils.psf import EPSFBuilder, extract_stars

    _, med, std = sigma_clipped_stats(data, sigma=3)
    finder = DAOStarFinder(fwhm=fwhm, threshold=20 * std, exclude_border=True)
    src = finder(data - med)
    if src is None or len(src) < 5:
        return None

    size = int(np.ceil(fwhm * 5)) | 1  # odd cutout size
    half = size // 2
    ny, nx = data.shape

    # keep bright, non-saturated, well-separated stars
    src = src[(src["peak"] < 0.85 * satur)]
    src.sort("flux"); src.reverse()
    xc, yc = _xy_cols(src)
    keep = []
    xs, ys = np.asarray(src[xc]), np.asarray(src[yc])
    for i, row in enumerate(src):
        x, y = float(row[xc]), float(row[yc])
        if x < half or y < half or x > nx - half or y > ny - half:
            continue
        d = np.hypot(xs - x, ys - y)
        if np.sum(d < 2 * size) > 1:   # crowded -> skip
            continue
        keep.append((x, y))
        if len(keep) >= 25:
            break
    if len(keep) < 5:
        return None

    stars_tbl = Table()
    stars_tbl["x"] = [k[0] for k in keep]
    stars_tbl["y"] = [k[1] for k in keep]
    nd = NDData(data=data - med)
    try:
        stars = extract_stars(nd, stars_tbl, size=size)
        builder = EPSFBuilder(oversampling=2, maxiters=8,
                              progress_bar=False, smoothing_kernel="quartic")
        epsf, _ = builder(stars)
        return epsf
    except Exception as exc:
        log(f"ePSF build failed: {exc}")
        return None


def aperture_photometry_at(data, err, positions, r, r_in, r_out):
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                     ApertureStats, aperture_photometry)
    positions = np.atleast_2d(positions)
    ap = CircularAperture(positions, r=r)
    ann = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    sky = ApertureStats(data, ann, sigma_clip=SigmaClip(sigma=3))
    phot = aperture_photometry(data, ap, error=err)
    bkg_per_pix = sky.median
    phot["aper_bkg"] = bkg_per_pix * ap.area
    phot["flux"] = phot["aperture_sum"] - phot["aper_bkg"]
    phot["flux_err"] = phot["aperture_sum_err"]
    return phot

def curve_of_growth_apcor(data, err, fwhm, sky, std, satur,
                          r_small, r_big, r_in, r_out):
    from photutils.detection import DAOStarFinder
    finder = DAOStarFinder(fwhm=fwhm, threshold=30 * std, exclude_border=True)
    src = finder(data - sky)
    if src is None or len(src) < 3:
        return 0.0, 0.05
    src = src[src["peak"] < 0.8 * satur]
    src.sort("flux"); src.reverse()
    src = src[: min(30, len(src))]
    xc, yc = _xy_cols(src)
    pos = np.transpose([np.asarray(src[xc]), np.asarray(src[yc])])
    small = aperture_photometry_at(data, err, pos, r_small, r_in, r_out)
    big = aperture_photometry_at(data, err, pos, r_big, r_in, r_out)
    good = (small["flux"] > 0) & (big["flux"] > 0)
    if good.sum() < 3:
        return 0.0, 0.05
    dmag = -2.5 * np.log10(small["flux"][good] / big["flux"][good])
    dmag = sigma_clip(dmag, sigma=3, maxiters=3)
    return float(np.ma.median(dmag)), float(np.ma.std(dmag))

def psf_forced_phot(data, err, epsf, x, y, fit_shape=7):
    if epsf is None:
        return np.nan, np.nan
    from photutils.psf import PSFPhotometry
    try:
        model = epsf.copy()
        # forced photometry: freeze the centroid, fit only the flux
        model.x_0.fixed = True
        model.y_0.fixed = True
        psfphot = PSFPhotometry(model, fit_shape=(fit_shape, fit_shape),
                                aperture_radius=fit_shape)
        init = Table({"x_0": [x], "y_0": [y]})
        out = psfphot(data, error=err, init_params=init)
        flux = float(out["flux_fit"][0])
        ferr = float(out["flux_err"][0]) if "flux_err" in out.colnames else np.nan
        return flux, ferr
    except Exception as exc:
        log(f"PSF photometry failed: {exc}")
        return np.nan, np.nan
    
# Gaia DR3 synthetic photometry calibration catalogue (built ONCE per run)

def _gspc_columns(band: str):
    """(mag, flux, flux_error) column names in synthetic_photometry_gspc."""
    if band in "ugriz":
        return (f"{band}_sdss_mag", f"{band}_sdss_flux", f"{band}_sdss_flux_error")
    if band in _JOHNSON:
        b = band.lower()
        return (f"{b}_jkc_mag", f"{b}_jkc_flux", f"{b}_jkc_flux_error")
    return None


def build_catalog_gspc(center: SkyCoord, radius_arcmin: float, bands, cfg: Config):
    from astroquery.gaia import Gaia

    systems = sorted({BAND_SYSTEM[b] for b in bands if b in BAND_SYSTEM})
    if not systems:
        log(f"No supported calibration system for bands {sorted(set(bands))}.")
        return Table()

    # request every band of each needed system (gives colour-pair partners too)
    sel_bands = []
    for system in systems:
        sel_bands += SYSTEM_BANDS[system]
    phot_cols = []
    for b in sel_bands:
        phot_cols += [f"s.{c}" for c in _gspc_columns(b)]

    gmax = min(cfg.cal_gmax, 17.65)   # GSPC is undefined fainter than this
    Gaia.ROW_LIMIT = -1
    adql = f"""
        SELECT TOP {int(cfg.cal_max_stars)}
               g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch,
               g.phot_g_mean_mag, s.c_star,
               {', '.join(phot_cols)}
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.synthetic_photometry_gspc AS s ON g.source_id = s.source_id
        WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                           CIRCLE('ICRS', {center.ra.deg}, {center.dec.deg},
                                  {radius_arcmin / 60.0}))
          AND g.phot_g_mean_mag BETWEEN {cfg.cal_gmin} AND {gmax}
          AND ABS(s.c_star) < {cfg.cstar_max}
        ORDER BY g.random_index
    """
    log(f"Querying GSPC synthetic photometry (r={radius_arcmin:.1f}', "
        f"G={cfg.cal_gmin}-{gmax}, |C*|<{cfg.cstar_max}, systems={systems}) ...")
    gaia = Gaia.launch_job_async(adql).get_results()
    if len(gaia) == 0:
        log("No GSPC calibrators in field; try a looser --cstar-max or fainter "
            "--cal-gmax, or --use-gaiaxpy.")
        return Table()
    log(f"  {len(gaia)} GSPC calibrators (no spectra fetched)")

    cols = {c.lower(): c for c in gaia.colnames}
    cat = Table()
    cat["source_id"] = gaia["source_id"]
    cat["ra"] = gaia["ra"]
    cat["dec"] = gaia["dec"]
    for c in ("pmra", "pmdec"):
        col = gaia[c]
        cat[c] = col.filled(0.0) if hasattr(col, "filled") else col
    cat["ref_epoch"] = gaia["ref_epoch"]

    for b in sel_bands:
        mcol, fcol, ecol = _gspc_columns(b)
        mname = cols.get(mcol.lower())
        if mname is None:
            continue
        mag = np.asarray(gaia[mname], dtype=float)
        magerr = np.full(len(cat), 0.02)
        fname, ename = cols.get(fcol.lower()), cols.get(ecol.lower())
        if fname and ename:
            f_ = np.asarray(gaia[fname], dtype=float)
            e_ = np.asarray(gaia[ename], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                magerr = np.where(f_ > 0, 1.0857 * e_ / f_, 0.02)
        cat[f"mag_{b}"] = mag
        cat[f"magerr_{b}"] = magerr
    return cat

def build_catalog_gaiaxpy(center: SkyCoord, radius_arcmin: float, bands, cfg: Config):
    """
    Fallback: takes long time
    """
    from astroquery.gaia import Gaia
    from gaiaxpy import generate, PhotometricSystem

    systems = sorted({BAND_SYSTEM[b] for b in bands if b in BAND_SYSTEM})
    if not systems:
        return Table()

    Gaia.ROW_LIMIT = -1
    adql = f"""
        SELECT TOP {int(cfg.cal_max_stars)}
               source_id, ra, dec, pmra, pmdec, ref_epoch, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE has_xp_continuous = 'true'
          AND phot_g_mean_mag BETWEEN {cfg.cal_gmin} AND {cfg.cal_gmax}
          AND 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {center.ra.deg}, {center.dec.deg},
                                  {radius_arcmin / 60.0}))
        ORDER BY random_index
    """
    log(f"Querying Gaia XP-continuous sources (GaiaXPy path, r={radius_arcmin:.1f}') ...")
    gaia = Gaia.launch_job_async(adql).get_results()
    if len(gaia) == 0:
        return Table()
    log(f"  {len(gaia)} sources -> integrating spectra with GaiaXPy (slow, once)")

    phot_systems = [getattr(PhotometricSystem, s) for s in systems]
    src_ids = list(np.asarray(gaia["source_id"], dtype=np.int64))
    synth = generate(src_ids, photometric_system=phot_systems, save_file=False)
    synth = Table.from_pandas(synth) if not isinstance(synth, Table) else synth
    cols = {c.lower(): c for c in synth.colnames}
    smap = {int(s): i for i, s in enumerate(synth["source_id"])}

    cat = Table()
    cat["source_id"] = gaia["source_id"]
    cat["ra"] = gaia["ra"]; cat["dec"] = gaia["dec"]
    for c in ("pmra", "pmdec"):
        col = gaia[c]
        cat[c] = col.filled(0.0) if hasattr(col, "filled") else col
    cat["ref_epoch"] = gaia["ref_epoch"]

    idx = np.array([smap.get(int(s), -1) for s in cat["source_id"]])
    for system in systems:
        label = SYSTEM_LABEL.get(system, system.capitalize())
        for b in SYSTEM_BANDS[system]:
            mcol = cols.get(f"{label}_mag_{b}".lower())
            fcol = cols.get(f"{label}_flux_{b}".lower())
            ecol = cols.get(f"{label}_flux_error_{b}".lower())
            if mcol is None:
                continue
            mag = np.full(len(cat), np.nan); magerr = np.full(len(cat), 0.02)
            for k, j in enumerate(idx):
                if j < 0:
                    continue
                mag[k] = float(synth[mcol][j])
                if fcol and ecol and synth[fcol][j]:
                    magerr[k] = float(1.0857 * synth[ecol][j] / synth[fcol][j])
            cat[f"mag_{b}"] = mag
            cat[f"magerr_{b}"] = magerr
    return cat

def build_calibration_catalog(center: SkyCoord, radius_arcmin: float,
                              bands, cfg: Config):
    """Dispatch to the fast GSPC query (default) or GaiaXPy spectra (fallback)."""
    if cfg.use_gaiaxpy:
        return build_catalog_gaiaxpy(center, radius_arcmin, bands, cfg)
    return build_catalog_gspc(center, radius_arcmin, bands, cfg)


def propagate_pm(cal: Table, obs_mjd: float) -> SkyCoord:
    ref_epoch = float(np.nanmedian(cal["ref_epoch"]))
    c = SkyCoord(ra=np.asarray(cal["ra"]) * u.deg,
                 dec=np.asarray(cal["dec"]) * u.deg,
                 pm_ra_cosdec=np.nan_to_num(np.asarray(cal["pmra"])) * u.mas / u.yr,
                 pm_dec=np.nan_to_num(np.asarray(cal["pmdec"])) * u.mas / u.yr,
                 obstime=Time(ref_epoch, format="jyear"))
    try:
        return c.apply_space_motion(new_obstime=Time(obs_mjd, format="mjd"))
    except Exception:
        return c  # if PM/epoch missing, use catalogue positions

# Zeropoint solution
def solve_zeropoint(sci: Frame, fwhm, sky, std, cat: Table, apcor,
                    band: str, cfg: Config):
    """
    Measure instrumental mags of Gaia calibrators on the science frame and fit
        mag_synth(band) = m_inst + ZP + k * colour
    where colour is the band's canonical colour pair (e.g. g-r for g/r). The
    synthetic mags come from the pre-built catalogue (no GaiaXPy call here).
    Returns dict with zp, zp_err, color_term, n_stars, rms.
    """
    magcol = f"mag_{band}"
    if magcol not in cat.colnames:
        log(f"No synthetic {band}-band magnitudes in catalogue.")
        return None

    err = make_error_map(sci.data, sky, std, sci.gain)
    r = cfg.aper_fwhm * fwhm
    r_in, r_out = cfg.sky_in_fwhm * fwhm, cfg.sky_out_fwhm * fwhm

    coords = propagate_pm(cat, sci.mjd)
    x, y = sci.wcs.world_to_pixel(coords)
    ny, nx = sci.data.shape
    inside = (x > r_out) & (y > r_out) & (x < nx - r_out) & (y < ny - r_out)
    inside &= np.isfinite(np.asarray(cat[magcol]))
    sub = cat[inside]; x = x[inside]; y = y[inside]
    if len(sub) < 3:
        return None

    pos = np.transpose([x, y])
    phot = aperture_photometry_at(sci.data, err, pos, r, r_in, r_out)
    flux = np.asarray(phot["flux"]); ferr = np.asarray(phot["flux_err"])
    good = (flux > 0) & (flux / np.maximum(ferr, 1e-9) > 20)
    if good.sum() < 3:
        return None

    m_inst = -2.5 * np.log10(flux[good]) + apcor
    mag_syn = np.asarray(sub[magcol])[good]
    mag_err = np.asarray(sub[f"magerr_{band}"])[good] if f"magerr_{band}" in sub.colnames \
        else np.full(good.sum(), 0.02)

    blue, red = COLOR_PAIR.get(band, (band, band))
    if f"mag_{blue}" in sub.colnames and f"mag_{red}" in sub.colnames:
        color = (np.asarray(sub[f"mag_{blue}"]) - np.asarray(sub[f"mag_{red}"]))[good]
        color = np.nan_to_num(color, nan=np.nanmedian(color))
        have_color = True
    else:
        color = np.zeros(good.sum()); have_color = False

    m_err = 1.0857 * ferr[good] / flux[good]
    w = 1.0 / np.sqrt(mag_err**2 + m_err**2 + 0.01**2)
    delta = mag_syn - m_inst  # = ZP + k*colour

    if cfg.fit_color_term and have_color and good.sum() >= 6:
        A = np.vstack([np.ones_like(color), color]).T
        for _ in range(3):                 # sigma-clipped weighted LSQ
            sol, *_ = np.linalg.lstsq(A * w[:, None], delta * w, rcond=None)
            resid = delta - A @ sol
            keep = np.abs(resid - np.median(resid)) < 3 * (np.std(resid) + 1e-9)
            A, delta, color, w = A[keep], delta[keep], color[keep], w[keep]
            if keep.all():
                break
        zp, k = float(sol[0]), float(sol[1])
        rms = float(np.std(delta - A @ sol)); n = len(delta)
    else:
        clipped = sigma_clip(delta, sigma=3, maxiters=5)
        zp = float(np.ma.average(clipped, weights=w[~clipped.mask]))
        k = 0.0
        rms = float(np.ma.std(clipped)); n = int(clipped.count())

    zp_err = rms / np.sqrt(max(n, 1))
    return dict(zp=zp, zp_err=zp_err, color_term=k, n_stars=n, rms=rms,
                color_pair=f"{blue}-{red}" if have_color else "none")


def make_error_map(data, sky, std, gain):
    from photutils.utils import calc_total_error
    bkg_err = np.full_like(data, std, dtype=np.float64)
    src = np.clip(data - sky, 0, None)
    return calc_total_error(src, bkg_err, effective_gain=max(gain, 1e-3))

def save_cutouts(sci, ref_aligned, diff, x, y, out_png, box=45,
                 aper_r=None, title=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from astropy.visualization import ZScaleInterval
    except Exception:
        return
    zs = ZScaleInterval()
    xi, yi = int(round(x)), int(round(y))
    ny, nx = sci.shape
    y0, y1 = max(yi - box, 0), min(yi + box, ny)
    x0, x1 = max(xi - box, 0), min(xi + box, nx)
    sl = (slice(y0, y1), slice(x0, x1))
    cx, cy = x - x0, y - y0                       # SN position within the cutout

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    panels = [(sci, "science"), (ref_aligned, "reference"), (diff, "difference")]
    for ax, (im, label) in zip(axes, panels):
        cut = np.asarray(im[sl], dtype=float)
        lo, hi = zs.get_limits(cut[np.isfinite(cut)] if np.isfinite(cut).any() else cut)
        ax.imshow(cut, vmin=lo, vmax=hi, cmap="gray", origin="lower")
        if aper_r:
            ax.add_patch(Circle((cx, cy), aper_r, ec="red", fc="none", lw=1.2))
        else:
            ax.plot(cx, cy, "r+", ms=14, mew=1.4)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)




def process_epoch(sci_path, ref: Frame, ref_fwhm, ref_sky, ref_std,
                  cat, band, cfg: Config):
    sci = load_frame(sci_path, cfg)
    log(f"=== {sci.name}  band={sci.band}->{band}  EXPTIME={sci.exptime:.1f}s "
        f"MJD={sci.mjd:.4f} ===")

    if cfg.flatten_bkg:
        sci_fwhm0, _, _, _ = measure_fwhm_bkg(sci.data)
        sci.data, struct = flatten_background(
            sci.data, cfg.bkg_box, cfg.bkg_filter, sci_fwhm0)
        log(f"flattened science background (removed structure rms={struct:.2f})")

    sci_fwhm, sci_sky, sci_std, _ = measure_fwhm_bkg(sci.data)
    log(f"science FWHM={sci_fwhm:.2f} px  sky={sci_sky:.1f}  rms={sci_std:.1f}")

    workdir = os.path.join(cfg.outdir, sci.name)
    os.makedirs(workdir, exist_ok=True)

    ref_aligned_path = os.path.join(workdir, "ref_aligned.fits")
    reproject_reference(ref, sci, ref_aligned_path)
    sci_plain = write_plain_fits(sci, os.path.join(workdir, "sci.fits"))
    ref_aligned = load_frame(ref_aligned_path, cfg)

    diff_path = os.path.join(workdir, "diff.fits")
    noise_path = os.path.join(workdir, "diff.noise.fits")
    conv_tmpl = run_hotpants(sci_plain, ref_aligned_path, diff_path, noise_path,
                             sci, ref_aligned, sci_fwhm, ref_fwhm, cfg)
    if conv_tmpl is False and not os.path.exists(diff_path):
        return None

    diff = fits.getdata(diff_path).astype(np.float64)
    diff_noise = (fits.getdata(noise_path).astype(np.float64)
                  if os.path.exists(noise_path)
                  else make_error_map(diff, 0.0, sci_std, sci.gain))
    diff_noise = np.where(np.isfinite(diff_noise) & (diff_noise > 0),
                          diff_noise, np.nanmedian(diff_noise))

    if conv_tmpl:
        epsf = build_epsf(sci.data, sci_sky, sci_fwhm, sci.saturate)
        psf_fwhm = sci_fwhm
    else:
        epsf = build_epsf(ref_aligned.data, ref_sky, ref_fwhm, ref.saturate)
        psf_fwhm = ref_fwhm

    sn = SkyCoord(cfg.ra * u.deg, cfg.dec * u.deg)
    xsn, ysn = sci.wcs.world_to_pixel(sn)
    xsn, ysn = float(xsn), float(ysn)

    r = cfg.aper_fwhm * sci_fwhm
    r_in, r_out = cfg.sky_in_fwhm * sci_fwhm, cfg.sky_out_fwhm * sci_fwhm
    apcor, apcor_err = curve_of_growth_apcor(
        sci.data, make_error_map(sci.data, sci_sky, sci_std, sci.gain),
        sci_fwhm, sci_sky, sci_std, sci.saturate,
        r, cfg.big_aper_fwhm * sci_fwhm, r_in, r_out)

    ap = aperture_photometry_at(diff, diff_noise, [xsn, ysn], r, r_in, r_out)
    flux_ap = float(ap["flux"][0]); ferr_ap = float(ap["flux_err"][0])
    flux_psf, ferr_psf = psf_forced_phot(diff, diff_noise, epsf, xsn, ysn,
                                         fit_shape=int(np.ceil(psf_fwhm * 1.5)) | 1)

    zp = solve_zeropoint(sci, sci_fwhm, sci_sky, sci_std, cat, apcor, band, cfg)
    if zp is None:
        log("WARNING: zeropoint failed (too few calibrators).")
        zp = dict(zp=np.nan, zp_err=np.nan, color_term=0.0, n_stars=0,
                  rms=np.nan, color_pair="none")

    def calibrate(flux, ferr):
        if not np.isfinite(flux) or flux <= 0:
            return np.nan, np.nan
        m_inst = -2.5 * np.log10(flux) + apcor
        m = m_inst + zp["zp"]
        if cfg.fit_color_term and cfg.sn_color is not None:
            m += zp["color_term"] * cfg.sn_color
        merr = np.sqrt((1.0857 * ferr / flux) ** 2 + zp["zp_err"] ** 2
                       + apcor_err ** 2)
        return m, merr

    mag_ap, mag_ap_err = calibrate(flux_ap, ferr_ap)
    mag_psf, mag_psf_err = calibrate(flux_psf, ferr_psf)


    noise_in_ap = float(np.sqrt(np.sum(
        diff_noise[_aperture_mask(diff.shape, xsn, ysn, r)] ** 2)))
    lim_flux = cfg.nsigma_limit * noise_in_ap
    limmag = (-2.5 * np.log10(lim_flux) + apcor + zp["zp"]
              if np.isfinite(zp["zp"]) and lim_flux > 0 else np.nan)

    snr_ap = flux_ap / ferr_ap if ferr_ap and ferr_ap > 0 else np.nan
    detected = np.isfinite(snr_ap) and snr_ap >= cfg.nsigma_limit

    if cfg.cutouts or cfg.plots:
        if np.isfinite(mag_ap):
            sub = f"{band}={mag_ap:.2f}+/-{mag_ap_err:.2f}  SNR={snr_ap:.1f}"
        else:
            sub = f"{band}>{limmag:.2f} ({cfg.nsigma_limit:.0f}sigma limit)"
        title = (f"{sci.name}   MJD={sci.mjd:.3f}   "
                 f"FWHM={sci_fwhm:.1f}px   {sub}")
        save_cutouts(sci.data, ref_aligned.data, diff, xsn, ysn,
                     os.path.join(workdir, f"{sci.name}_sci_ref_diff.png"),
                     aper_r=r, title=title)

    return dict(
        frame=sci.name, band=band, raw_filter=sci.band, mjd=sci.mjd,
        exptime=sci.exptime, fwhm_pix=sci_fwhm,
        zp=zp["zp"], zp_err=zp["zp_err"], color_term=zp["color_term"],
        color_pair=zp.get("color_pair", "none"),
        n_cal=zp["n_stars"], zp_rms=zp["rms"],
        x_sn=xsn, y_sn=ysn, apcor=apcor,
        flux_ap=flux_ap, flux_ap_err=ferr_ap, snr_ap=snr_ap,
        mag_ap=mag_ap, mag_ap_err=mag_ap_err,
        flux_psf=flux_psf, flux_psf_err=ferr_psf,
        mag_psf=mag_psf, mag_psf_err=mag_psf_err,
        detected=bool(detected),
        limit_mag=limmag, nsigma=cfg.nsigma_limit,
    )


def _aperture_mask(shape, x, y, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2






# Parallel execution

_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

_WCAT = None
_WCFG = None


def _init_worker(cat, cfg):
    global _WCAT, _WCFG
    _WCAT, _WCFG = cat, cfg
    for v in _THREAD_ENV:
        os.environ.setdefault(v, "1")


def _process_one(task):
    sci_path, ref_path, ref_fwhm, ref_sky, ref_std, band = task
    try:
        ref = load_frame(ref_path, _WCFG)
        res = process_epoch(sci_path, ref, ref_fwhm, ref_sky, ref_std,
                            _WCAT, band, _WCFG)
        return res if res else {"_skip": os.path.basename(sci_path)}
    except Exception as exc:
        import traceback
        return {"_error": f"{os.path.basename(sci_path)}: {exc}",
                "_tb": traceback.format_exc()}


def _write_ref_fits(ref: Frame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = ref.wcs.to_header()
    hdr["EXPTIME"] = ref.exptime
    hdr["GAIN"] = ref.gain
    hdr["RDNOISE"] = ref.rdnoise
    hdr["SATURATE"] = ref.saturate
    hdr["FILTER"] = ref.band
    fits.writeto(path, np.asarray(ref.data, np.float32), hdr, overwrite=True)
    return path


# Main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sci", required=True,
                   help="folder of science images (any band) -- or a glob")
    p.add_argument("--ref", required=True,
                   help="folder of reference images (one per band) -- or a file")
    p.add_argument("--ra", type=float, required=True, help="SN RA (deg)")
    p.add_argument("--dec", type=float, required=True, help="SN Dec (deg)")
    p.add_argument("--outdir", default="out")
    p.add_argument("--hotpants", default="hotpants")
    p.add_argument("--bands", default=None,
                   help="comma-list to restrict processing, e.g. 'g,r,i'")
    p.add_argument("--cal-max-stars", type=int, default=200,
                   help="max calibrators (default 200; one fast GSPC query)")
    p.add_argument("--cal-gmin", type=float, default=13.0,
                   help="bright Gaia G cut for calibrators")
    p.add_argument("--cal-gmax", type=float, default=17.5,
                   help="faint Gaia G cut (GSPC reliable to ~17.65)")
    p.add_argument("--cstar-max", type=float, default=0.10,
                   help="GSPC |C*| quality cut (smaller = cleaner, fewer stars)")
    p.add_argument("--use-gaiaxpy", action="store_true",
                   help="integrate BP/RP spectra with GaiaXPy instead of GSPC (slow)")
    p.add_argument("--cal-radius", type=float, default=0.0,
                   help="Gaia cone radius in arcmin (0 = auto from frame)")
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--rdnoise", type=float, default=None)
    p.add_argument("--saturate", type=float, default=None)
    p.add_argument("--saturate-frac", type=float, default=0.90,
                   help="HOTPANTS upper data limit as a fraction of saturation")
    p.add_argument("--kernel-order", type=int, default=2,
                   help="HOTPANTS spatial kernel order (-ko)")
    p.add_argument("--no-flatten-bkg", action="store_true",
                   help="disable large-scale background flattening")
    p.add_argument("--bkg-box", type=int, default=128,
                   help="Background2D mesh size in px (keep LARGE for hosts)")
    p.add_argument("--bkg-order", type=int, default=1,
                   help="HOTPANTS spatial background order (-bgo)")
    p.add_argument("--sn-color", type=float, default=None,
                   help="SN colour (this band's pair) for colour-term correction")
    p.add_argument("--no-color-term", action="store_true")
    p.add_argument("--nsigma-limit", type=float, default=5.0)
    p.add_argument("--ncpu", type=int, default=1,
                   help="parallel workers over epochs (default 1 = serial)")
    p.add_argument("--no-cutouts", action="store_true",
                   help="disable the per-epoch sci/ref/diff triptych")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args(argv)

    def gather(path):
        if os.path.isdir(path):
            files = []
            for ext in ("*.fits", "*.fits.fz", "*.fz", "*.fit"):
                files += glob.glob(os.path.join(path, ext))
            return sorted(files)
        return sorted(glob.glob(path))

    sci_files = gather(args.sci)
    if not sci_files:
        log("No science files matched."); sys.exit(1)

    cfg = Config(
        sci_glob=args.sci, ref_path=args.ref, ra=args.ra, dec=args.dec,
        outdir=args.outdir, hotpants=args.hotpants,
        gain=args.gain, rdnoise=args.rdnoise, saturate=args.saturate,
        saturate_frac=args.saturate_frac, kernel_order=args.kernel_order,
        flatten_bkg=not args.no_flatten_bkg, bkg_box=args.bkg_box,
        bkg_order=args.bkg_order, cal_max_stars=args.cal_max_stars,
        cal_radius_arcmin=args.cal_radius, cal_gmin=args.cal_gmin,
        cal_gmax=args.cal_gmax, cstar_max=args.cstar_max,
        use_gaiaxpy=args.use_gaiaxpy,
        fit_color_term=not args.no_color_term, sn_color=args.sn_color,
        nsigma_limit=args.nsigma_limit, plots=args.plots,
        ncpu=max(1, args.ncpu), cutouts=not args.no_cutouts,
    )
    os.makedirs(cfg.outdir, exist_ok=True)

    only = {b.strip() for b in args.bands.split(",")} if args.bands else None
    sci_by_band: dict[str, list] = {}
    for f in sci_files:
        b, raw = peek_band(f)
        if b is None:
            log(f"skip (unknown filter {raw!r}): {os.path.basename(f)}"); continue
        if only and b not in only:
            continue
        sci_by_band.setdefault(b, []).append(f)
    if not sci_by_band:
        log("No science frames with a recognised, requested band."); sys.exit(1)
    log("Science bands: " + ", ".join(f"{b}({len(v)})"
                                      for b, v in sci_by_band.items()))

    ref_by_band: dict[str, str] = {}
    if os.path.isdir(args.ref):
        for rf in gather(args.ref):
            rb, _ = peek_band(rf)
            if rb and rb not in ref_by_band:
                ref_by_band[rb] = rf
    else:
        rb, _ = peek_band(args.ref)
        ref_by_band[rb if rb else "_single"] = args.ref
    log("Reference bands: " + (", ".join(ref_by_band) or "none"))

    # --- build the Gaia synthetic calibration catalogue ONCE ---
    sn = SkyCoord(cfg.ra * u.deg, cfg.dec * u.deg)
    if cfg.cal_radius_arcmin > 0:
        radius = cfg.cal_radius_arcmin
    else:
        probe = load_frame(sci_files[0], cfg)
        ny, nx = probe.data.shape
        corners = probe.wcs.pixel_to_world([0, nx - 1, 0, nx - 1],
                                           [0, 0, ny - 1, ny - 1])
        radius = float(np.max(sn.separation(corners).arcmin)) + 1.0
    log("Building Gaia DR3 XP synthetic catalogue once for the whole run ...")
    cat = build_calibration_catalog(sn, radius, list(sci_by_band.keys()), cfg)
    if len(cat) == 0:
        log("WARNING: empty calibration catalogue; magnitudes will be NaN.")


    refs_dir = os.path.join(cfg.outdir, "_refs")
    ref_cache = {}
    for band in sci_by_band:
        ref_src = ref_by_band.get(band) or ref_by_band.get("_single")
        if ref_src is None:
            log(f"No reference for band '{band}' -> skipping "
                f"{len(sci_by_band[band])} frames.")
            continue
        log(f"--- band {band}: reference {os.path.basename(ref_src)} ---")
        ref = load_frame(ref_src, cfg)
        if cfg.flatten_bkg:
            rf0, _, _, _ = measure_fwhm_bkg(ref.data)
            ref.data, struct = flatten_background(ref.data, cfg.bkg_box,
                                                  cfg.bkg_filter, rf0)
            log(f"flattened reference background (structure rms={struct:.2f})")
        ref_fwhm, ref_sky, ref_std, _ = measure_fwhm_bkg(ref.data)
        ref.band = band
        ref_flat = _write_ref_fits(ref, os.path.join(refs_dir, f"ref_{band}.fits"))
        ref_cache[band] = (ref_flat, ref_fwhm, ref_sky, ref_std)
        log(f"reference FWHM={ref_fwhm:.2f} px")

    tasks = []
    for band, files in sci_by_band.items():
        if band not in ref_cache:
            continue
        rp, rf, rs, rstd = ref_cache[band]
        for f in files:
            tasks.append((f, rp, rf, rs, rstd, band))
    if not tasks:
        log("No epochs to process (missing references?)."); return

    rows = []

    def _handle(res):
        if not res or "_skip" in res:
            return
        if "_error" in res:
            log("FAILED " + res["_error"])
            print(res.get("_tb", ""), file=sys.stderr)
            return
        rows.append(res)
        b, m = res["band"], res["mag_ap"]
        if np.isfinite(m):
            tag = "DET" if res["detected"] else f"<{cfg.nsigma_limit:.0f}sig"
            log(f"  done {res['frame']}: {b}={m:.3f}+/-{res['mag_ap_err']:.3f} "
                f"[{tag}] ZP={res['zp']:.3f} n_cal={res['n_cal']}")
        else:
            log(f"  done {res['frame']}: {b}>{res['limit_mag']:.2f} "
                f"({cfg.nsigma_limit:.0f}sigma limit)")

    if cfg.ncpu > 1 and len(tasks) > 1:
        for v in _THREAD_ENV:
            os.environ.setdefault(v, "1")
        nproc = min(cfg.ncpu, len(tasks))
        log(f"Processing {len(tasks)} epochs on {nproc} workers ...")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(cat, cfg)) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for fut in as_completed(futures):
                _handle(fut.result())
    else:
        log(f"Processing {len(tasks)} epochs serially ...")
        _init_worker(cat, cfg)
        for t in tasks:
            _handle(_process_one(t))

    if rows:
        tbl = Table(rows)
        tbl.sort(["band", "mjd"])
        out_csv = os.path.join(cfg.outdir, "lightcurve.csv")
        tbl.write(out_csv, format="csv", overwrite=True)
        for band in sorted(set(tbl["band"])):
            sub = tbl[tbl["band"] == band]
            sub.write(os.path.join(cfg.outdir, f"lightcurve_{band}.csv"),
                      format="csv", overwrite=True)
        log(f"Light curve written: {out_csv}  "
            f"({len(tbl)} epochs across {len(set(tbl['band']))} band(s))")
    else:
        log("No epochs produced output.")


if __name__ == "__main__":
    main()