# noawclg

**noawclg** is a Python library for downloading and analysing atmospheric and ocean data
from NOAA operational models — GFS weather forecasts and GODAS/ERSST ocean analyses.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

examples/index
gallery/index
```

```{toctree}
:maxdepth: 3
:caption: API reference

api/index
```

---

## Highlights

| Feature | Details |
|---------|---------|
| **GFS forecasts** | 3-hourly, 0–384 h, global, surface + multi-level |
| **Ocean temperature** | GODAS 40 levels, 5 m – 4 478 m, 1980–present |
| **Salinity** | GODAS `salt`, PSU, full water column |
| **Ocean currents** | GODAS `ucur` + `vcur`, speed + direction |
| **Sea surface height** | GODAS `sshg`, geoid-relative, key ENSO indicator |
| **SST record** | ERSST v5 back to 1854 — long climatologies |
| **ENSO diagnostics** | ONI, Niño indices, phase classification, D20, WWV |
| **Plotting** | 26 ready-to-use plot functions — maps, globe, ENSO, GFS |
| **No API key** | All data accessed via public OPeNDAP / NOMADS |

## Quick install

```bash
pip install noawclg
# with plotting extras
pip install "noawclg[plots]"
```

## Links

- [GitHub](https://github.com/reinanbr/noawclg)
- [PyPI](https://pypi.org/project/noawclg/)
- [Issues](https://github.com/reinanbr/noawclg/issues)
