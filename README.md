# Spectral Analysis Utilities

This repository contains scripts and example data for experimenting with NDVI and spectral reflectance models. It is organised into four main directories:

- **data/** – input datasets such as spectral CSV files, pigment spectra and a ZIP archive of published leaf litter measurements.
- **scripts/** – Python utilities to fit filter curves, simulate NDVI and mix reflectance spectra.
- **figures/** – figures produced by the scripts.
- **results/** – generated files such as JSON coefficient tables and bundled ZIP archives.

## Data

The `data` directory stores several reference spectra:

- `SERAPH_R118_absorbtion (1).json` – transmission profile of an optical filter.
- `chl_a_122_abs.txt` and `chl_b_125_abs.txt` – chlorophyll absorption data from PhotochemCAD.
- `intact_spec.csv`, `ground_spec.csv`, and `healthy_leaf.csv` – example reflectance measurements.
- `2012-leaf-reflectance-spectra-of-tropical-trees-in-tapajos-national-forest.csv` – healthy leaf spectra.
- `doi_10_5061_dryad_hdr7sqvrk__v20240426.zip` – archive containing additional litter spectra and metadata.
- `dataSpec_P5.csv` – soil spectra.

## Scripts

The `scripts` folder contains several Python programs:

- `bundle.py` – downloads pigment spectra, extracts reference data and plots filter transmission alongside chlorophyll reflectance. It creates `spectra_bundle.zip` and `transmission_vs_chl_reflect.png` in the `results` and `figures` directories.
- `r118_filter_fit.py` – fits high degree polynomials to the R118 filter transmission curve and saves the coefficients to `results/poly14_trans_fit.json`. Various plots of the fits and residues are saved to `figures/`.
- `simulate_ndvi.py` – computes NDVI values for each row in a reflectance CSV file using the supplied red and NIR filter profiles.
- `simulate_filtered_reflectance.py` – mixes leaf litter, healthy leaf and ground spectra in different proportions and reports the sensor reading after applying a filter profile.
- `simulate_mixed_ndvi.py` – similar to the above but outputs NDVI instead of the raw sensor value.

All scripts assume they are executed from the repository root so that relative paths to the `data` directory work out of the box. Each script prints usage information when run with `-h`.

## Figures and Results

Running the scripts will create PNG figures in `figures/` and additional JSON or ZIP artifacts in `results/`. Several example outputs are already included for reference.

## Usage Example

Install the required packages:

```bash
pip install numpy matplotlib requests pandas scipy
```

Then run one of the scripts, for example:

```bash
python scripts/simulate_ndvi.py --csv data/intact_spec.csv
```

## Licence

The example data are reproduced from their respective sources. The scripts in this repository are released under the MIT licence.
