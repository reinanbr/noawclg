# Plotting

All plot functions live in `plots.py` at the project root (not inside the
`noawclg` package), so they can be used independently.

```python
from plots import plot_synoptic_map, plot_enso_index, plot_globe
```

## GFS / atmospheric plots

```{eval-rst}
.. autofunction:: plots.plot_synoptic_map
.. autofunction:: plots.plot_timeseries
.. autofunction:: plots.plot_wind_rose
.. autofunction:: plots.plot_hovmoller
.. autofunction:: plots.plot_precip_heatmap
.. autofunction:: plots.plot_spread_matrix
```

## Ocean / ENSO plots

```{eval-rst}
.. autofunction:: plots.plot_enso_index
.. autofunction:: plots.plot_ocean_temp_map
.. autofunction:: plots.plot_thermocline_section
.. autofunction:: plots.plot_ssh_map
.. autofunction:: plots.plot_ocean_currents
.. autofunction:: plots.plot_globe
```
