O pedido é bem direto — editar o README para marcar as keys bloqueadas na tabela do Variable Catalogue. Não preciso criar um arquivo Word, só preciso identificar onde essas keys aparecem no documento e indicar que são `pgrb2b` apenas.

Aqui está o README atualizado com as keys bloqueadas marcadas na tabela do Variable Catalogue:

---

### Surface / single-level variables — alterações

As seguintes keys da tabela de superfície precisam ser marcadas como **pgrb2b only** (não disponíveis no endpoint `pgrb2` padrão):

| Key | Long name | Units | Nota |
|-----|-----------|-------|------|
| `t2m` | 2 metre temperature | °C | ✅ pgrb2 |
| `d2m` | 2 metre dewpoint temperature | °C | ✅ pgrb2 |
| `r2` | 2 metre relative humidity | % | ✅ pgrb2 |
| `sh2` | 2 metre specific humidity | kg kg⁻¹ | ✅ pgrb2 |
| **`aptmp`** | Apparent temperature | °C | ⛔ pgrb2b only |
| `u10` | 10 metre U wind component | m s⁻¹ | ✅ pgrb2 |
| `v10` | 10 metre V wind component | m s⁻¹ | ✅ pgrb2 |
| `gust` | Wind speed (gust) | m s⁻¹ | ✅ pgrb2 |
| `prmsl` | Pressure reduced to MSL | hPa | ✅ pgrb2 |
| `mslet` | MSLP (Eta model reduction) | hPa | ✅ pgrb2 |
| `sp` | Surface pressure | hPa | ✅ pgrb2 |
| `orog` | Orography | m | ✅ pgrb2 |
| `lsm` | Land-sea mask | 0–1 | ✅ pgrb2 |
| `vis` | Visibility | m | ✅ pgrb2 |
| `prate` | Precipitation rate | kg m⁻² s⁻¹ | ✅ pgrb2 |
| `cpofp` | Percent frozen precipitation | % | ✅ pgrb2 |
| `crain` | Categorical rain | — | ✅ pgrb2 |
| `csnow` | Categorical snow | — | ✅ pgrb2 |
| `cfrzr` | Categorical freezing rain | — | ✅ pgrb2 |
| `cicep` | Categorical ice pellets | — | ✅ pgrb2 |
| `sde` | Snow depth | m | ✅ pgrb2 |
| `sdwe` | Water equivalent of snow depth | kg m⁻² | ✅ pgrb2 |
| `pwat` | Precipitable water | kg m⁻² | ✅ pgrb2 |
| `cwat` | Cloud water | kg m⁻² | ✅ pgrb2 |
| `tcc` | Total cloud cover | % | ✅ pgrb2 |
| **`lcc`** | Low cloud cover | % | ⛔ pgrb2b only |
| **`mcc`** | Medium cloud cover | % | ⛔ pgrb2b only |
| **`hcc`** | High cloud cover | % | ⛔ pgrb2b only |
| `lftx` | Surface lifted index | K | ✅ pgrb2 |
| `lftx4` | Best (4-layer) lifted index | K | ✅ pgrb2 |
| **`hlcy`** | Storm relative helicity | m² s⁻² | ⛔ heightAboveGroundLayer (endpoint diferente) |
| `refc` | Composite radar reflectivity | dB | ✅ pgrb2 |
| `siconc` | Sea ice area fraction | 0–1 | ✅ pgrb2 |
| `veg` | Vegetation | % | ✅ pgrb2 |
| **`tozne`** | Total ozone | DU | ⛔ f000 only |

### Multi-level variables — alterações

| Key | Long name | Units | Nota |
|-----|-----------|-------|------|
| **`t`** | Temperature | °C | ⛔ pgrb2b only |
| **`r`** | Relative humidity | % | ⛔ pgrb2b only |
| **`q`** | Specific humidity | kg kg⁻¹ | ⛔ pgrb2b only |
| **`gh`** | Geopotential height | gpm | ⛔ pgrb2b only |
| **`u`** | U component of wind | m s⁻¹ | ⛔ pgrb2b only |
| **`v`** | V component of wind | m s⁻¹ | ⛔ pgrb2b only |
| **`w`** | Vertical velocity | Pa s⁻¹ | ⛔ pgrb2b only |
| **`absv`** | Absolute vorticity | s⁻¹ | ⛔ pgrb2b only |
| `cape` | CAPE | J kg⁻¹ | ✅ pgrb2 |
| `cin` | Convective inhibition | J kg⁻¹ | ✅ pgrb2 |
| **`st`** | Soil temperature | °C | ⛔ depthBelowLandLayer (não exposto) |
| **`soilw`** | Volumetric soil moisture | Proportion | ⛔ depthBelowLandLayer (não exposto) |

---

Também vale adicionar uma nota de aviso no topo da seção Variable Catalogue, algo assim:

> **⚠️ Atenção:** As keys marcadas com ⛔ **não funcionam** com o endpoint `pgrb2` padrão usado por `GFSDatasetManager`. Tentativas de usá-las resultam em HTTP 500. Veja `UNSUPPORTED_KEYS` no código-fonte.

Quer que eu gere o README completo atualizado como arquivo para download?