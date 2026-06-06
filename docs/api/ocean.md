# Ocean data — GODAS & ERSST

```{eval-rst}
.. automodule:: noawclg.ocean
   :members:
   :undoc-members:
   :show-inheritance:
```

## Constants

```{eval-rst}
.. autodata:: noawclg.GODAS_VARS
   :annotation:

   Dictionary mapping GODAS variable names to metadata:

   .. code-block:: python

      {
          "pottmp": {"long_name": "Potential temperature",   "units_out": "°C",  ...},
          "salt":   {"long_name": "Salinity",                "units_out": "PSU", ...},
          "ucur":   {"long_name": "U-current (eastward)",    "units_out": "m/s", ...},
          "vcur":   {"long_name": "V-current (northward)",   "units_out": "m/s", ...},
          "sshg":   {"long_name": "Sea Surface Height",      "units_out": "m",   ...},
      }

.. autodata:: noawclg.NINO_BOXES
   :annotation:

   Standard ENSO monitoring boxes (longitude in 0–360 convention):

   .. code-block:: python

      {
          "1+2": {"lat": (-10,  0), "lon": (270, 280)},
          "3":   {"lat": ( -5,  5), "lon": (210, 270)},
          "3.4": {"lat": ( -5,  5), "lon": (190, 240)},
          "4":   {"lat": ( -5,  5), "lon": (160, 210)},
      }
```

## GODAS access

```{eval-rst}
.. autofunction:: noawclg.open_godas
.. autofunction:: noawclg.get_godas
.. autofunction:: noawclg.get_ocean_temp
.. autofunction:: noawclg.get_salinity
.. autofunction:: noawclg.get_currents
.. autofunction:: noawclg.get_ssh
```

## ERSST v5

```{eval-rst}
.. autofunction:: noawclg.open_ersst
```

## ENSO diagnostics

```{eval-rst}
.. autofunction:: noawclg.get_sst_series
.. autofunction:: noawclg.get_nino_anomaly
.. autofunction:: noawclg.get_oni
.. autofunction:: noawclg.classify_enso
.. autofunction:: noawclg.get_thermocline_depth
.. autofunction:: noawclg.get_warm_water_volume
.. autofunction:: noawclg.enso_summary
```
