# Installation

## Requirements

- Python 3.10+
- pip

## Basic install

```bash
pip install noawclg
```

## With plotting extras

The plotting module (`plots.py`) requires optional heavy dependencies.
Install them all at once:

```bash
pip install "noawclg[plots]"
```

This adds:

| Package | Used for |
|---------|---------|
| `matplotlib` | All plots |
| `cartopy` | Maps, synoptic charts, globe view |
| `scipy` | Smoothing, contour interpolation |
| `seaborn` | Heatmaps, spread matrices |
| `windrose` | Wind-rose diagrams |
| `metpy` | SkewT, meteorological calculations |
| `cmocean` | Scientific ocean/atmosphere colourmaps |

## Development install

```bash
git clone https://github.com/reinanbr/noawclg
cd noawclg
pip install -e ".[dev]"
```

## Verifying the install

```python
import noawclg
print(noawclg.__version__)
print(list(noawclg.GODAS_VARS.keys()))
# ['pottmp', 'salt', 'ucur', 'vcur', 'sshg']
```

## Optional: eccodes (GRIB support)

GFS data is served in GRIB2 format.  noawclg uses `cfgrib` which requires
the **eccodes** C library.  On most systems it is installed automatically
as a Python wheel.  If you see `eccodes not found`, install it manually:

```bash
# Ubuntu / Debian
sudo apt install libeccodes-dev

# macOS
brew install eccodes

# conda
conda install -c conda-forge eccodes
```
