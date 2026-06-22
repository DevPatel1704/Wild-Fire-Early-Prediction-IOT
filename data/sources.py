"""Layer 1 data sources: ERA5 reanalysis, NFDB fire records, FIRMS/NASA hotspots.

Each loader reads a real file from data/raw if present; otherwise it raises so
build_dataset.py can fall back to the synthetic generator. The synthetic path
lets the whole pipeline run end-to-end with no downloads, then you swap in real
files later without touching any other code.

Real-data download pointers (see README):
  ERA5  : https://cds.climate.copernicus.eu  -> NetCDF, daily, BC bbox
  NFDB  : https://cwfis.cfs.nrcan.gc.ca       -> fire point shapefile / CSV
  FIRMS : https://firms.modaps.eosdis.nasa.gov -> active-fire CSV
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from config import DATA_DIR, BC_BOUNDS, TOP_10_FEATURES


# --------------------------------------------------------------------------- #
# Real loaders                                                                #
# --------------------------------------------------------------------------- #
def load_era5(path=None) -> pd.DataFrame:
    """ERA5 NetCDF -> tidy DataFrame [date, lat, lon, <features...>].

    Requires `xarray` + `netCDF4`. Variable names are mapped onto the paper's
    top-10 codes; adjust the rename map to your CDS request.
    """
    path = path or DATA_DIR / "era5.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    import xarray as xr
    ds = xr.open_dataset(path)
    df = ds.to_dataframe().reset_index()
    rename = {  # CDS short names -> our feature codes (edit to match your pull)
        "swvl1": "swvl1", "swvl2": "swvl2", "mn2t": "mn2t", "blh": "blh",
        "pev": "pev", "time": "date", "latitude": "lat", "longitude": "lon",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def load_nfdb(path=None) -> pd.DataFrame:
    """NFDB fire points -> [date, lat, lon]. Accepts CSV or shapefile."""
    path = path or DATA_DIR / "nfdb.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
    else:
        import geopandas as gpd
        g = gpd.read_file(path).to_crs(4326)
        df = pd.DataFrame({"lat": g.geometry.y, "lon": g.geometry.x,
                           "date": g.get("REP_DATE")})
    cols = {c.lower(): c for c in df.columns}
    return df.rename(columns={cols.get("latitude", "lat"): "lat",
                              cols.get("longitude", "lon"): "lon"})[["date", "lat", "lon"]]


def load_firms(path=None) -> pd.DataFrame:
    """FIRMS active-fire CSV -> [date, lat, lon]."""
    path = path or DATA_DIR / "firms.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df.rename(columns={"acq_date": "date", "latitude": "lat",
                              "longitude": "lon"})[["date", "lat", "lon"]]


# --------------------------------------------------------------------------- #
# Synthetic fallback (runs with zero downloads)                               #
# --------------------------------------------------------------------------- #
def synthesize(n_days=365 * 2, grid=20, seed=7) -> pd.DataFrame:
    """Generate a physically-plausible ERA5-like grid + fire labels.

    Fire probability is driven by the same physics the paper highlights: dry
    soil, hot/dry air, strong wind, summer seasonality. This makes the learned
    model behaviour sane even on synthetic data.
    """
    rng = np.random.default_rng(seed)
    lats = np.linspace(BC_BOUNDS["lat_min"], BC_BOUNDS["lat_max"], grid)
    lons = np.linspace(BC_BOUNDS["lon_min"], BC_BOUNDS["lon_max"], grid)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")

    rows = []
    for d in dates:
        doy = d.dayofyear
        season = math.sin((doy - 80) / 365 * 2 * math.pi)  # peak ~ summer
        for la in lats:
            for lo in lons:
                base_t = 5 + 18 * season + rng.normal(0, 3)
                swvl1 = np.clip(0.35 - 0.18 * season + rng.normal(0, 0.05), 0, 1)
                swvl2 = np.clip(swvl1 + rng.normal(0, 0.03), 0, 1)
                lgws = abs(rng.normal(4 + 2 * season, 2))
                mgws = lgws + abs(rng.normal(2, 1))
                pev = max(0, rng.normal(2 + 3 * season, 1))
                blh = max(50, rng.normal(800 + 600 * season, 200))
                gwd = rng.uniform(0, 360)
                vilwd = rng.normal(0, 1)
                rows.append([d, la, lo, swvl1, base_t, lgws, pev, doy,
                             gwd, blh, mgws, vilwd, swvl2])

    cols = ["date", "lat", "lon"] + TOP_10_FEATURES
    # mn2t expects temperature; reorder to match TOP_10_FEATURES exactly
    df = pd.DataFrame(rows, columns=["date", "lat", "lon", "swvl1", "mn2t",
                                     "lgws", "pev", "DOY", "gwd", "blh",
                                     "mgws", "vilwd", "swvl2"])

    # ground-truth fire driver -> probability -> label (intercept tuned for a
    # realistic but learnable ~8-12% positive rate on synthetic data)
    z = (-3.2
         - 7.0 * df["swvl1"]
         + 0.12 * df["mn2t"]
         + 0.10 * df["lgws"]
         + 0.20 * df["pev"]
         + 1.5 * np.sin((df["DOY"] - 80) / 365 * 2 * math.pi))
    p = 1 / (1 + np.exp(-z))
    df["fire"] = (rng.random(len(df)) < p).astype(int)
    return df
