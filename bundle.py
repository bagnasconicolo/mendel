#!/usr/bin/env python3
"""
plot_transmission_vs_chl_reflectance.py
---------------------------------------
Overlay a normalised transmission spectrum with the *reflectance*
curves of chlorophyll‐a and chlorophyll‐b (350–950 nm).

Creates:
  spectra_bundle.zip
    ├─ SERAPH_R118_absorbtion (1).json       (your data)
    ├─ chl_a_122_abs.txt  (downloaded)       (PhotochemCAD ID 122)
    ├─ chl_b_125_abs.txt  (downloaded)       (PhotochemCAD ID 125)
    └─ transmission_vs_chl_reflect.png       (the figure)

Requires:
    pip install numpy matplotlib requests pandas scipy
"""
from pathlib import Path
import json, zipfile, requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# ── inputs & URLs ────────────────────────────────────────────────────
json_path   = Path("SERAPH_R118_absorbtion (1).json")
url_chl_a   = "https://omlc.org/spectra/PhotochemCAD/data/122-abs.txt"
url_chl_b   = "https://omlc.org/spectra/PhotochemCAD/data/125-abs.txt"
file_chl_a  = Path("chl_a_122_abs.txt")
file_chl_b  = Path("chl_b_125_abs.txt")

# Local copy of litter spectra archive (provided by user)
leaf_zip_path = Path("doi_10_5061_dryad_hdr7sqvrk__v20240426.zip")
file_leaf     = Path("intact_spec.csv")
file_healthy  = Path("2012-leaf-reflectance-spectra-of-tropical-trees-in-tapajos-national-forest.csv")

# Soil spectra (dry & wet) provided by user
file_soil     = Path("dataSpec_P5.csv")     # uploaded CSV

zip_out     = Path("spectra_bundle.zip")
wl_min, wl_max = 350, 950   # nm
# ─────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    if not dest.exists():
        print(f"⇣ {dest.name} …", end="", flush=True)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print("done")
    else:
        print(f"✓ {dest.name} already here")


# Helper: try each URL in list until one succeeds, save to dest, else raise
def download_first_available(urls: list[str], dest: Path) -> None:
    """
    Try each URL in the provided list until one succeeds.
    Saves the content to *dest*. Raises a RuntimeError if all URLs fail.
    """
    if dest.exists():
        print(f"✓ {dest.name} already here")
        return
    for url in urls:
        try:
            print(f"⇣ {dest.name} from {url} …", end="", flush=True)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print("done")
            return
        except Exception:
            print("failed")
    raise RuntimeError(f"None of the candidate URLs for {dest.name} are reachable.")


def extract_leaf_csv(zip_path: Path, csv_name: str, dest: Path) -> None:
    """Extract csv_name from zip_path into dest if not already present."""
    if dest.exists():
        print(f"✓ {dest.name} already here")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected {zip_path} not found; please place the litter data ZIP alongside this script")
    print(f"⇣ extracting {csv_name} from {zip_path.name} …", end="", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(csv_name, path=".")
    print("done")


def read_photochemcad(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wl, eps = [], []
    for line in txt_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            w, e = map(float, line.split()[:2])
            wl.append(w); eps.append(e)
    wl, eps = np.array(wl), np.array(eps)
    # keep 350–950 nm window (but pad to 950 if needed)
    mask = (wl >= wl_min) & (wl <= 950)
    wl, eps = wl[mask], eps[mask]
    if wl[-1] < wl_max:          # extend flat out to 950 nm
        pad_wl  = np.arange(wl[-1]+1, wl_max+1, 1)
        pad_eps = np.full_like(pad_wl, eps[-1])
        wl      = np.concatenate((wl, pad_wl))
        eps     = np.concatenate((eps, pad_eps))
    return wl, eps


def normalise(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def _load_soil_csv_threecol(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a CSV that contains soil spectra for dry and wet samples.
    Two supported layouts:

    1. COLUMN layout (λ, dry, wet) ... as three numeric columns.
    2. ROW layout  (first row contains wavelengths, later rows labelled
       'dry soil' and 'wet soil') like dataSpec_P5.csv.

    Returns (wavelength_nm, dry_reflectance, wet_reflectance)
    """
    # --- Try simple 3-column numeric layout first ----------------------------
    try:
        arr = np.loadtxt(path, delimiter=",")
        if arr.shape[1] >= 3:
            wl, dry, wet = arr[:, 0], arr[:, 1], arr[:, 2]
            return wl, dry, wet
    except Exception:
        pass  # fall through to row-oriented parser

    # --- Row-oriented layout --------------------------------------------------
    df = pd.read_csv(path, header=None, na_filter=False)

    # Identify wavelength row
    if not str(df.iloc[0, 0]).lower().startswith("wavelength"):
        raise ValueError("Unexpected soil CSV format (no 'Wavelength' header)")

    wl = df.iloc[0, 1:].astype(float).to_numpy()

    # Find rows whose first cell contains 'dry soil' / 'wet soil'
    row_dry = df[df.iloc[:, 0].str.contains("dry soil", case=False, na=False)]
    row_wet = df[df.iloc[:, 0].str.contains("wet soil", case=False, na=False)]
    if row_dry.empty or row_wet.empty:
        raise ValueError("Could not locate 'dry soil' and 'wet soil' rows in CSV")

    dry = row_dry.iloc[0, 1:].astype(float).to_numpy()
    wet = row_wet.iloc[0, 1:].astype(float).to_numpy()

    return wl, dry, wet


# ─────────────────────────────────────────────────────────────────────
# 1) download pigment files ----------------------------------------------------
download(url_chl_a, file_chl_a)
download(url_chl_b, file_chl_b)
extract_leaf_csv(leaf_zip_path, "intact_spec.csv", file_leaf)

# 2) load your transmission JSON ------------------------------------------------
with json_path.open() as f:
    raw = json.load(f)

pairs        = raw["datasetColl"][0]["data"]
wl_tr        = np.fromiter((p["value"][0] for p in pairs), float)
tr           = np.fromiter((p["value"][1] for p in pairs), float)

# ── NEW: drop the last sample point ───────────────────────────────────────────
wl_tr, tr    = wl_tr[:-1], tr[:-1]          # <-- this line removes the tail

tr_n         = normalise(tr)

# 3) load & process Chl-a / Chl-b ---------------------------------------------
wl_a, eps_a  = read_photochemcad(file_chl_a)
wl_b, eps_b  = read_photochemcad(file_chl_b)

ref_a        = 1 - normalise(eps_a)      # absorption → reflectance
ref_b        = 1 - normalise(eps_b)

# Interpolate both pigments onto a common 1-nm grid (350–950 nm)
wl_common    = np.arange(wl_min, wl_max + 1, 1)
ref_a_i      = np.interp(wl_common, wl_a, ref_a)
ref_b_i      = np.interp(wl_common, wl_b, ref_b)

# 3b) create smoothed sum (Chl-a + Chl-b) curve and renormalise to full scale
sum_raw      = (gaussian_filter1d(ref_a_i, sigma=3) + gaussian_filter1d(ref_b_i, sigma=3)) / 2
sum_ref      = normalise(sum_raw)

# 4) parse the intact leaf-litter CSV: wavelengths are column headers 400–2400
leaf_df     = pd.read_csv(file_leaf)
wl_cols     = [c for c in leaf_df.columns if c.isdigit()]
wl_leaf     = np.array(list(map(int, wl_cols)))
refl_matrix = leaf_df[wl_cols].to_numpy(dtype=float)
refl_dead   = np.nanmean(refl_matrix, axis=0)  # mean spectrum across samples

# keep 350–950 nm window
m_leaf      = (wl_leaf >= wl_min) & (wl_leaf <= wl_max)
wl_leaf     = wl_leaf[m_leaf]
refl_dead   = refl_dead[m_leaf]
refl_leaf_n = normalise(refl_dead)

# 5) optionally parse the ground spectrum CSV: (if present)
file_ground = Path("ground_spec.csv")
if file_ground.exists():
    ground_df     = pd.read_csv(file_ground)
    wl_cols_g     = [c for c in ground_df.columns if c.isdigit()]
    wl_ground     = np.array(list(map(int, wl_cols_g)))
    refl_matrix_g = ground_df[wl_cols_g].to_numpy(dtype=float)
    refl_ground   = np.nanmean(refl_matrix_g, axis=0)  # mean spectrum across samples

    # keep 350–950 nm window
    m_ground      = (wl_ground >= wl_min) & (wl_ground <= wl_max)
    wl_ground     = wl_ground[m_ground]
    refl_ground   = refl_ground[m_ground]
    refl_ground_n = normalise(refl_ground)
else:
    wl_ground, refl_ground_n = np.array([]), np.array([])

# 6) Healthy leaf spectrum (Tapajós dataset uploaded by user) --------------
try:
    df_h = pd.read_csv(file_healthy)
    # Detect wavelength column
    if "wavelength" in df_h.columns:
        wl_h      = df_h["wavelength"].to_numpy(dtype=float)
        refl_cols = [c for c in df_h.columns if c not in ("wavelength", "species", "sampleID")]
        refl_mat  = df_h[refl_cols].to_numpy(dtype=float)
        refl_h    = np.nanmean(refl_mat, axis=1)
    else:
        # assume first numeric columns across header are wavelengths
        num_cols  = [c for c in df_h.columns if str(c).replace(".", "", 1).isdigit()]
        wl_h      = np.array(list(map(float, num_cols)))
        refl_h    = np.nanmean(df_h[num_cols].to_numpy(dtype=float), axis=0)

    # clip 350–950 nm
    mh           = (wl_h >= wl_min) & (wl_h <= wl_max)
    wl_health    = wl_h[mh]
    refl_health  = refl_h[mh]
    refl_health_n = normalise(refl_health)
except Exception as e:
    print("WARNING: Could not read healthy leaf CSV:", e)
    wl_health, refl_health_n = np.array([]), np.array([])

# 7) Soil reflectance spectra (dry and wet) -----------------------------------
try:
    wl_soil_full, refl_dry_full, refl_wet_full = _load_soil_csv_threecol(file_soil)
    mask_soil = (wl_soil_full >= wl_min) & (wl_soil_full <= wl_max)

    wl_soil  = wl_soil_full[mask_soil]
    refl_dry = refl_dry_full[mask_soil]
    refl_wet = refl_wet_full[mask_soil]

    refl_dry_n = normalise(refl_dry)
    refl_wet_n = normalise(refl_wet)
except Exception as e:
    print("WARNING: Could not parse soil CSV:", e)
    wl_soil, refl_dry_n, refl_wet_n = np.array([]), np.array([]), np.array([])

# 8) plot ----------------------------------------------------------------------
colors = [
    "#252525",  # gray for R118 Transmittance
    "#c20389",  # magenta for Chl-a+b
    "#ff7f0e",  # orange for leaf litter
    "#2ca02c",  # green for healthy leaf
]

plt.figure(figsize=(9,5))
plt.plot(wl_tr, tr_n, linewidth=3, color=colors[0], label="R118 Transmittance")
plt.plot(wl_common, sum_ref, "--", linewidth=1, color=colors[1], label="Chlorophyll‐a+b")

if wl_leaf.size:
    plt.plot(wl_leaf, refl_leaf_n, "-", linewidth=3, color=colors[2], label="Leaf‐litter reflectance")

if wl_health.size:
    plt.plot(wl_health, refl_health_n, "-", linewidth=3, color=colors[3], label="Healthy leaf reflectance")

if wl_soil.size:

    plt.plot(wl_soil, refl_wet_n, "-", linewidth=3, color="#FF0000", label="Wet Soil")

plt.xlim(wl_min, wl_max)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalised intensity")
plt.title("R118 Filter Transmittance and NDVI-Relevant Reflectance Spectra")
plt.legend()
plt.tight_layout()

fig_name = Path("transmission_vs_chl_reflect.png")
plt.savefig(fig_name, dpi=300)

# Optionally zoom x‐axis and show interactively
plt.xlim(400, 850)
plt.show()

