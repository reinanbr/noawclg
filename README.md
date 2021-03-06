
<h1 align='center'>NOAWClg</h1>
<p align='center'>

<br/>
<a href="https://github.com/perseu912"><img title="Autor" src="https://img.shields.io/badge/Autor-reinan_br-blue.svg?style=for-the-badge&logo=github"></a>
<br/>
<a href='http://dgp.cnpq.br/dgp/espelhogrupo/0180330616769073'><img src='https://shields.io/badge/cnpq-grupo_de_fisica_computacional_ifsertao--pe-blueviolet?logo=appveyor&style=for-the-badge'></a>

<p align='center'>
<!-- github dados --
<!-- sites de pacotes -->
<a href='https://pypi.org/project/noaawc/'><img src='https://img.shields.io/pypi/v/noawclg'></a>
<a href='#'><img src='https://img.shields.io/pypi/wheel/noawclg'></a>
<a href='#'><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/noawclg"></a>
<img alt="PyPI - License" src="https://img.shields.io/pypi/l/noawclg">
<br/>
<!-- outros premios e analises -->
<!-- <a href='#'><img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/perseu912/noawclg?logo=codefactor">
</a> -->
<!-- redes sociais -->
<a href='https://instagram.com/gpftc_ifsertao/'><img src='https://shields.io/badge/insta-gpftc_ifsertao-darkviolet?logo=instagram&style=flat'></a>
<a href='https://discord.gg/pFZP86gvEm'><img src='https://img.shields.io/discord/856582838467952680.svg?label=discord&logo=discord'></a>

</p>
</p>
<p align='center'> <b>Library for getting  the world data climate from the data noaa/nasa</b></p>
<hr/>

## Instalation

```sh
$ pip3 install noawcgl -U
```

## Examples
### getting data
<br>

#### from a point
getting the data:
```py
from noawclg import get_noaa_data as gnd

point = (-9.41,-40.5)

data = gnd.get_data_from_point(point)

# a example for the surface temperature
data = {'time':data['time'],'data':data['tmpsfc']}

print(data)
```

```sh
{'time': <xarray.IndexVariable 'time' (time: 129)>
array(['2022-01-01T00:00:00.000000000', '2022-01-01T03:00:00.000000000',
       '2022-01-01T06:00:00.000000000', '2022-01-01T09:00:00.000000000',
       '2022-01-01T12:00:00.000000000', '2022-01-01T15:00:00.000000000',
       '2022-01-01T18:00:00.000000000', '2022-01-01T21:00:00.000000000',
       '2022-01-02T00:00:00.000000000', '2022-01-02T03:00:00.000000000',
       '2022-01-02T06:00:00.000000000', '2022-01-02T09:00:00.000000000',
       '2022-01-02T12:00:00.000000000', '2022-01-02T15:00:00.000000000',
       '2022-01-02T18:00:00.000000000', '2022-01-02T21:00:00.000000000',
       '2022-01-03T00:00:00.000000000', '2022-01-03T03:00:00.000000000',
                                    ....
       '2022-01-15T18:00:00.000000000', '2022-01-15T21:00:00.000000000',
       '2022-01-16T00:00:00.000000000', '2022-01-16T03:00:00.000000000',
       '2022-01-16T06:00:00.000000000', '2022-01-16T09:00:00.000000000',
       '2022-01-16T12:00:00.000000000', '2022-01-16T15:00:00.000000000',
       '2022-01-16T18:00:00.000000000', '2022-01-16T21:00:00.000000000',
       '2022-01-17T00:00:00.000000000'], dtype='datetime64[ns]')
Attributes:
    grads_dim:      t
    grads_mapping:  linear
    grads_size:     129
    grads_min:      00z01jan2022
    grads_step:     3hr
    long_name:      time
    minimum:        00z01jan2022
    maximum:        00z17jan2022
    resolution:     0.125, 'data': <xarray.Variable (time: 129)>
array([302.32916, 302.33063, 302.31046, 302.7827 , 303.02896, 302.28903,
       302.29962, 302.29337, 302.25876, 302.302  , 302.41086, 302.907  ,
       303.10004, 302.4816 , 302.3765 , 302.38263, 302.3673 , 302.40558,
       302.4181 , 303.05243, 303.3014 , 302.52554, 302.30768, 302.2898 ,
       302.27194, 302.25403, 302.44955, 303.30478, 303.53696, 302.60095,
       302.3012 , 302.30118, 302.28815, 302.34244, 302.37427, 303.0273 ,
       303.33054, 302.54422, 302.4121 , 302.35107, 302.3121 , 302.35263,
       302.44736, 302.88632, 303.20844, 302.51526, 302.4    , 302.31686,
       302.3237 , 302.36264, 302.3221 , 302.6572 , 302.64008, 302.30173,
       302.26715, 302.3487 , 302.2893 , 302.3021 , 302.372  , 302.79086,
       302.5877 , 302.3137 , 302.35422, 302.33835, 302.33206, 302.2572 ,
       302.28955, 302.70654, 302.62338, 302.33896, 302.2855 , 302.2695 ,
       302.28635, 302.30844, 302.28635, 302.69318, 302.7695 , 302.28632,
       302.2542 , 302.29767, 302.29282, 302.3    , 302.34424, 302.45758,
       302.6297 , 302.28122, 302.30002, 302.27756, 302.2916 , 302.28403,
       302.34793, 302.6237 , 302.7041 , 302.20798, 302.3054 , 302.31668,
       302.2472 , 302.3143 , 302.30505, 302.42892, 302.58902, 302.2641 ,
       302.30127, 302.31564, 302.30002, 302.31393, 302.3112 , 302.714  ,
       303.03915, 302.4548 , 302.3548 , 302.31393, 302.26282, 302.30002,
       302.3594 , 302.96558, 302.692  , 302.28314, 302.20352, 302.28723,
       302.23373, 302.26645, 302.30002, 302.6699 , 302.9055 , 302.4652 ,
       302.3585 , 302.26627, 302.30002], dtype=float32)
Attributes:
    long_name:  ** surface temperature [k] }
```

#### plot data from a place:
```py
from noawclg.main import get_noaa_data as gnd
data_noaa = gnd(gfs='1p00')

place = 'juazeiro BA'
print(data_noaa.get_noaa_keys())


## rain's (mm)
def fmt(data): return data* 100_000
data_noaa.plot_data_from_place(place=place,path_file='plot_ch1.png',
                               title='Previs??o de Chuvas\nPetrolina-PE/Juazeiro-BA',
                                ylabel='mm',fmt_data=fmt,key_noaa='prateavesfc')



## wind speed v-component (m/s)
def fmt_t(data): return data
data_noaa.plot_data_from_place(place=place,path_file='plot_wind100m.png',
                               title='Velocidade dos Ventos\nPetrolina-PE/Juazeiro-BA',
                                ylabel='m/s',fmt_data=fmt_t,key_noaa='vgrdmwl')

```
<hr>


### getting the list keys for get data (use it as guide)

```py
from noawclg import get_data_noaa as gdn

data_noaa = gdn()
print(data_noaa.get_noaa_keys())
```

```sh
[{'time': 'time'} {'lev': 'altitude'} {'lat': 'latitude'}
 {'lon': 'longitude'}
 {'absvprs': '** (1000 975 950 925 900.. 10 7 4 2 1) absolute vorticity [1/s] '}
 {'no4lftxsfc': '** surface best (4 layer) lifted index [k] '}
 {'acpcpsfc': '** surface convective precipitation [kg/m^2] '}
 {'albdosfc': '** surface albedo [%] '}
 {'apcpsfc': '** surface total precipitation [kg/m^2] '}
 {'capesfc': '** surface convective available potential energy [j/kg] '}
 {'cape180_0mb': '** 180-0 mb above ground convective available potential energy [j/kg] '}
 {'cape90_0mb': '** 90-0 mb above ground convective available potential energy [j/kg] '}
 {'cape255_0mb': '** 255-0 mb above ground convective available potential energy [j/kg] '}
 {'cfrzravesfc': '** surface categorical freezing rain [-] '}
 {'cfrzrsfc': '** surface categorical freezing rain [-] '}
 {'cicepavesfc': '** surface categorical ice pellets [-] '}
 {'cicepsfc': '** surface categorical ice pellets [-] '}
 {'cinsfc': '** surface convective inhibition [j/kg] '}
 {'cin180_0mb': '** 180-0 mb above ground convective inhibition [j/kg] '}
 {'cin90_0mb': '** 90-0 mb above ground convective inhibition [j/kg] '}
 {'cin255_0mb': '** 255-0 mb above ground convective inhibition [j/kg] '}
 {'clwmrprs': '** (1000 975 950 925 900.. 250 200 150 100 50) cloud mixing ratio [kg/kg] '}
 {'clwmrhy1': '** 1 hybrid level cloud mixing ratio [kg/kg] '}
 {'cnwatsfc': '** surface plant canopy surface water [kg/m^2] '}
 {'cpofpsfc': '** surface percent frozen precipitation [%] '}
 {'cpratavesfc': '** surface convective precipitation rate [kg/m^2/s] '}
 {'cpratsfc': '** surface convective precipitation rate [kg/m^2/s] '}
 {'crainavesfc': '** surface categorical rain [-] '}
 {'crainsfc': '** surface categorical rain [-] '}
 {'csnowavesfc': '** surface categorical snow [-] '}
 {'csnowsfc': '** surface categorical snow [-] '}
 {'cwatclm': '** entire atmosphere (considered as a single layer) cloud water [kg/m^2] '}
 {'cworkclm': '** entire atmosphere (considered as a single layer) cloud work function [j/kg] '}
 {'dlwrfsfc': '** surface downward long-wave rad. flux [w/m^2] '}
 {'dpt2m': '** 2 m above ground dew point temperature [k] '}
 {'dswrfsfc': '** surface downward short-wave radiation flux [w/m^2] '}
 {'dzdtprs': '** (1000 975 950 925 900.. 10 7 4 2 1) vertical velocity (geometric) [m/s] '}
 {'fldcpsfc': '** surface field capacity [fraction] '}
 {'fricvsfc': '** surface frictional velocity [m/s] '}
 {'gfluxsfc': '** surface ground heat flux [w/m^2] '}
 {'grleprs': '** (1000 975 950 925 900.. 250 200 150 100 50) graupel [kg/kg] '}
 {'grlehy1': '** 1 hybrid level graupel [kg/kg] '}
 {'gustsfc': '** surface wind speed (gust) [m/s] '}
 {'hcdcavehcll': '** high cloud layer high cloud cover [%] '}
 {'hcdchcll': '** high cloud layer high cloud cover [%] '}
 {'hgtsfc': '** surface geopotential height [gpm] '}
 {'hgtprs': '** (1000 975 950 925 900.. 10 7 4 2 1) geopotential height [gpm] '}
 {'hgt2pv': '** pv=2e-06 (km^2/kg/s) surface geopotential height [gpm] '}
 {'hgtneg2pv': '** pv=-2e-06 (km^2/kg/s) surface geopotential height [gpm] '}
 {'hgttop0c': '** highest tropospheric freezing level geopotential height [gpm] '}
 {'hgtceil': '** cloud ceiling geopotential height [gpm] '}
 {'hgt0c': '** 0c isotherm geopotential height [gpm] '}
 {'hgtmwl': '** max wind geopotential height [gpm] '}
 {'hgttrop': '** tropopause geopotential height [gpm] '}
 {'hindexsfc': '** surface haines index [numeric] '}
 {'hlcy3000_0m': '** 3000-0 m above ground storm relative helicity [m^2/s^2] '}
 {'hpblsfc': '** surface planetary boundary layer height [m] '}
 {'icahtmwl': '** max wind icao standard atmosphere reference height [m] '}
 {'icahttrop': '** tropopause icao standard atmosphere reference height [m] '}
 {'icecsfc': '** surface ice cover [proportion] '}
 {'iceg_10m': '** 10 m above mean sea level ice growth rate [m/s] '}
 {'icetksfc': '** surface ice thickness [m] '}
 {'icetmpsfc': '** surface ice temperature [k] '}
 {'icmrprs': '** (1000 975 950 925 900.. 250 200 150 100 50) ice water mixing ratio [kg/kg] '}
 {'icmrhy1': '** 1 hybrid level ice water mixing ratio [kg/kg] '}
 {'landsfc': '** surface land cover (0=sea, 1=land) [proportion] '}
 {'lcdcavelcll': '** low cloud layer low cloud cover [%] '}
 {'lcdclcll': '** low cloud layer low cloud cover [%] '}
 {'lftxsfc': '** surface surface lifted index [k] '}
 {'lhtflsfc': '** surface latent heat net flux [w/m^2] '}
 {'mcdcavemcll': '** middle cloud layer medium cloud cover [%] '}
 {'mcdcmcll': '** middle cloud layer medium cloud cover [%] '}
 {'msletmsl': '** mean sea level mslp (eta model reduction) [pa] '}
 {'o3mrprs': '** (1000 975 950 925 900.. 10 7 4 2 1) ozone mixing ratio [kg/kg] '}
 {'pevprsfc': '** surface potential evaporation rate [w/m^2] '}
 {'plpl255_0mb': '** 255-0 mb above ground pressure of level from which parcel was lifted [pa] '}
 {'potsig995': '** 0.995 sigma level potential temperature [k] '}
 {'prateavesfc': '** surface precipitation rate [kg/m^2/s] '}
 {'pratesfc': '** surface precipitation rate [kg/m^2/s] '}
 {'preslclb': '** low cloud bottom level pressure [pa] '}
 {'preslclt': '** low cloud top level pressure [pa] '}
 {'presmclb': '** middle cloud bottom level pressure [pa] '}
 {'presmclt': '** middle cloud top level pressure [pa] '}
 {'preshclb': '** high cloud bottom level pressure [pa] '}
 {'preshclt': '** high cloud top level pressure [pa] '}
 {'pressfc': '** surface pressure [pa] '}
 {'pres80m': '** 80 m above ground pressure [pa] '}
 {'pres2pv': '** pv=2e-06 (km^2/kg/s) surface pressure [pa] '}
 {'presneg2pv': '** pv=-2e-06 (km^2/kg/s) surface pressure [pa] '}
 {'prescclb': '** convective cloud bottom level pressure [pa] '}
 {'prescclt': '** convective cloud top level pressure [pa] '}
 {'presmwl': '** max wind pressure [pa] '}
 {'prestrop': '** tropopause pressure [pa] '}
 {'prmslmsl': '** mean sea level pressure reduced to msl [pa] '}
 {'pwatclm': '** entire atmosphere (considered as a single layer) precipitable water [kg/m^2] '}
 {'refcclm': '** entire atmosphere composite reflectivity [db] '}
 {'refd4000m': '** 4000 m above ground reflectivity [db] '}
 {'refd1000m': '** 1000 m above ground reflectivity [db] '}
 {'refdhy1': '** 1 hybrid level reflectivity [db] '}
 {'refdhy2': '** 2 hybrid level reflectivity [db] '}
 {'rhprs': '** (1000 975 950 925 900.. 10 7 4 2 1) relative humidity [%] '}
 {'rh2m': '** 2 m above ground relative humidity [%] '}
 {'rhsg330_1000': '** 0.33-1 sigma layer relative humidity [%] '}
 {'rhsg440_1000': '** 0.44-1 sigma layer relative humidity [%] '}
 {'rhsg720_940': '** 0.72-0.94 sigma layer relative humidity [%] '}
 {'rhsg440_720': '** 0.44-0.72 sigma layer relative humidity [%] '}
 {'rhsig995': '** 0.995 sigma level relative humidity [%] '}
 {'rh30_0mb': '** 30-0 mb above ground relative humidity [%] '}
 {'rhclm': '** entire atmosphere (considered as a single layer) relative humidity [%] '}
 {'rhtop0c': '** highest tropospheric freezing level relative humidity [%] '}
 {'rh0c': '** 0c isotherm relative humidity [%] '}
 {'rwmrprs': '** (1000 975 950 925 900.. 250 200 150 100 50) rain mixing ratio [kg/kg] '}
 {'rwmrhy1': '** 1 hybrid level rain mixing ratio [kg/kg] '}
 {'sfcrsfc': '** surface surface roughness [m] '}
 {'shtflsfc': '** surface sensible heat net flux [w/m^2] '}
 {'snmrprs': '** (1000 975 950 925 900.. 250 200 150 100 50) snow mixing ratio [kg/kg] '}
 {'snmrhy1': '** 1 hybrid level snow mixing ratio [kg/kg] '}
 {'snodsfc': '** surface snow depth [m] '}
 {'soill0_10cm': '** 0-0.1 m below ground liquid volumetric soil moisture (non frozen) [proportion] '}
 {'soill10_40cm': '** 0.1-0.4 m below ground liquid volumetric soil moisture (non frozen) [proportion] '}
 {'soill40_100cm': '** 0.4-1 m below ground liquid volumetric soil moisture (non frozen) [proportion] '}
 {'soill100_200cm': '** 1-2 m below ground liquid volumetric soil moisture (non frozen) [proportion] '}
 {'soilw0_10cm': '** 0-0.1 m below ground volumetric soil moisture content [fraction] '}
 {'soilw10_40cm': '** 0.1-0.4 m below ground volumetric soil moisture content [fraction] '}
 {'soilw40_100cm': '** 0.4-1 m below ground volumetric soil moisture content [fraction] '}
 {'soilw100_200cm': '** 1-2 m below ground volumetric soil moisture content [fraction] '}
 {'sotypsfc': '** surface soil type [-] '}
 {'spfhprs': '** (1000 975 950 925 900.. 10 7 4 2 1) specific humidity [kg/kg] '}
 {'spfh2m': '** 2 m above ground specific humidity [kg/kg] '}
 {'spfh80m': '** 80 m above ground specific humidity [kg/kg] '}
 {'spfh30_0mb': '** 30-0 mb above ground specific humidity [kg/kg] '}
 {'sunsdsfc': '** surface sunshine duration [s] '}
 {'tcdcaveclm': '** entire atmosphere total cloud cover [%] '}
 {'tcdcblcll': '** boundary layer cloud layer total cloud cover [%] '}
 {'tcdcclm': '** entire atmosphere total cloud cover [%] '}
 {'tcdcprs': '** (1000 975 950 925 900.. 250 200 150 100 50) total cloud cover [%] '}
 {'tcdcccll': '** convective cloud layer total cloud cover [%] '}
 {'tmax2m': '** 2 m above ground maximum temperature [k] '}
 {'tmin2m': '** 2 m above ground minimum temperature [k] '}
 {'tmplclt': '** low cloud top level temperature [k] '}
 {'tmpmclt': '** middle cloud top level temperature [k] '}
 {'tmphclt': '** high cloud top level temperature [k] '}
 {'tmpsfc': '** surface temperature [k] '}
 {'tmpprs': '** (1000 975 950 925 900.. 10 7 4 2 1) temperature [k] '}
 {'tmp_1829m': '** 1829 m above mean sea level temperature [k] '}
 {'tmp_2743m': '** 2743 m above mean sea level temperature [k] '}
 {'tmp_3658m': '** 3658 m above mean sea level temperature [k] '}
 {'tmp2m': '** 2 m above ground temperature [k] '}
 {'tmp80m': '** 80 m above ground temperature [k] '}
 {'tmp100m': '** 100 m above ground temperature [k] '}
 {'tmpsig995': '** 0.995 sigma level temperature [k] '}
 {'tmp30_0mb': '** 30-0 mb above ground temperature [k] '}
 {'tmp2pv': '** pv=2e-06 (km^2/kg/s) surface temperature [k] '}
 {'tmpneg2pv': '** pv=-2e-06 (km^2/kg/s) surface temperature [k] '}
 {'tmpmwl': '** max wind temperature [k] '}
 {'tmptrop': '** tropopause temperature [k] '}
 {'tozneclm': '** entire atmosphere (considered as a single layer) total ozone [du] '}
 {'tsoil0_10cm': '** 0-0.1 m below ground soil temperature validation to deprecate [k] '}
 {'tsoil10_40cm': '** 0.1-0.4 m below ground soil temperature validation to deprecate [k] '}
 {'tsoil40_100cm': '** 0.4-1 m below ground soil temperature validation to deprecate [k] '}
 {'tsoil100_200cm': '** 1-2 m below ground soil temperature validation to deprecate [k] '}
 {'ugwdsfc': '** surface zonal flux of gravity wave stress [n/m^2] '}
 {'uflxsfc': '** surface momentum flux, u-component [n/m^2] '}
 {'ugrdprs': '** (1000 975 950 925 900.. 10 7 4 2 1) u-component of wind [m/s] '}
 {'ugrd_1829m': '** 1829 m above mean sea level u-component of wind [m/s] '}
 {'ugrd_2743m': '** 2743 m above mean sea level u-component of wind [m/s] '}
 {'ugrd_3658m': '** 3658 m above mean sea level u-component of wind [m/s] '}
 {'ugrd10m': '** 10 m above ground u-component of wind [m/s] '}
 {'ugrd20m': '** 20 m above ground u-component of wind [m/s] '}
 {'ugrd30m': '** 30 m above ground u-component of wind [m/s] '}
 {'ugrd40m': '** 40 m above ground u-component of wind [m/s] '}
 {'ugrd50m': '** 50 m above ground u-component of wind [m/s] '}
 {'ugrd80m': '** 80 m above ground u-component of wind [m/s] '}
 {'ugrd100m': '** 100 m above ground u-component of wind [m/s] '}
 {'ugrdsig995': '** 0.995 sigma level u-component of wind [m/s] '}
 {'ugrd30_0mb': '** 30-0 mb above ground u-component of wind [m/s] '}
 {'ugrd2pv': '** pv=2e-06 (km^2/kg/s) surface u-component of wind [m/s] '}
 {'ugrdneg2pv': '** pv=-2e-06 (km^2/kg/s) surface u-component of wind [m/s] '}
 {'ugrdpbl': '** planetary boundary layer u-component of wind [m/s] '}
 {'ugrdmwl': '** max wind u-component of wind [m/s] '}
 {'ugrdtrop': '** tropopause u-component of wind [m/s] '}
 {'ulwrfsfc': '** surface upward long-wave rad. flux [w/m^2] '}
 {'ulwrftoa': '** top of atmosphere upward long-wave rad. flux [w/m^2] '}
 {'ustm6000_0m': '** 6000-0 m above ground u-component storm motion [m/s] '}
 {'uswrfsfc': '** surface upward short-wave radiation flux [w/m^2] '}
 {'uswrftoa': '** top of atmosphere upward short-wave radiation flux [w/m^2] '}
 {'vgwdsfc': '** surface meridional flux of gravity wave stress [n/m^2] '}
 {'vegsfc': '** surface vegetation [%] '}
 {'vflxsfc': '** surface momentum flux, v-component [n/m^2] '}
 {'vgrdprs': '** (1000 975 950 925 900.. 10 7 4 2 1) v-component of wind [m/s] '}
 {'vgrd_1829m': '** 1829 m above mean sea level v-component of wind [m/s] '}
 {'vgrd_2743m': '** 2743 m above mean sea level v-component of wind [m/s] '}
 {'vgrd_3658m': '** 3658 m above mean sea level v-component of wind [m/s] '}
 {'vgrd10m': '** 10 m above ground v-component of wind [m/s] '}
 {'vgrd20m': '** 20 m above ground v-component of wind [m/s] '}
 {'vgrd30m': '** 30 m above ground v-component of wind [m/s] '}
 {'vgrd40m': '** 40 m above ground v-component of wind [m/s] '}
 {'vgrd50m': '** 50 m above ground v-component of wind [m/s] '}
 {'vgrd80m': '** 80 m above ground v-component of wind [m/s] '}
 {'vgrd100m': '** 100 m above ground v-component of wind [m/s] '}
 {'vgrdsig995': '** 0.995 sigma level v-component of wind [m/s] '}
 {'vgrd30_0mb': '** 30-0 mb above ground v-component of wind [m/s] '}
 {'vgrd2pv': '** pv=2e-06 (km^2/kg/s) surface v-component of wind [m/s] '}
 {'vgrdneg2pv': '** pv=-2e-06 (km^2/kg/s) surface v-component of wind [m/s] '}
 {'vgrdpbl': '** planetary boundary layer v-component of wind [m/s] '}
 {'vgrdmwl': '** max wind v-component of wind [m/s] '}
 {'vgrdtrop': '** tropopause v-component of wind [m/s] '}
 {'vissfc': '** surface visibility [m] '}
 {'vratepbl': '** planetary boundary layer ventilation rate [m^2/s] '}
 {'vstm6000_0m': '** 6000-0 m above ground v-component storm motion [m/s] '}
 {'vvelprs': '** (1000 975 950 925 900.. 10 7 4 2 1) vertical velocity (pressure) [pa/s] '}
 {'vvelsig995': '** 0.995 sigma level vertical velocity (pressure) [pa/s] '}
 {'vwsh2pv': '** pv=2e-06 (km^2/kg/s) surface vertical speed shear [1/s] '}
 {'vwshneg2pv': '** pv=-2e-06 (km^2/kg/s) surface vertical speed shear [1/s] '}
 {'vwshtrop': '** tropopause vertical speed shear [1/s] '}
 {'watrsfc': '** surface water runoff [kg/m^2] '}
 {'weasdsfc': '** surface water equivalent of accumulated snow depth [kg/m^2] '}
 {'wiltsfc': '** surface wilting point [fraction] '}
 {'var00212m': '** 2 m above ground desc [unit] '}]
```


#### getting the all data
```py
from noawclg import get_noaa_data as gnd

data_noaa = gnd()

print(data_noaa)
```

```sh

Frozen({'time': <xarray.IndexVariable 'time' (time: 129)>
array(['2022-01-01T00:00:00.000000000', '2022-01-01T03:00:00.000000000',
       '2022-01-01T06:00:00.000000000', '2022-01-01T09:00:00.000000000',
       '2022-01-01T12:00:00.000000000', '2022-01-01T15:00:00.000000000',
       '2022-01-01T18:00:00.000000000', '2022-01-01T21:00:00.000000000',
       '2022-01-02T00:00:00.000000000', '2022-01-02T03:00:00.000000000',
       '2022-01-02T06:00:00.000000000', '2022-01-02T09:00:00.000000000',
       '2022-01-02T12:00:00.000000000', '2022-01-02T15:00:00.000000000',
       '2022-01-02T18:00:00.000000000', '2022-01-02T21:00:00.000000000',
       '2022-01-03T00:00:00.000000000', '2022-01-03T03:00:00.000000000',
       '2022-01-03T06:00:00.000000000', '2022-01-03T09:00:00.000000000',
       '2022-01-03T12:00:00.000000000', '2022-01-03T15:00:00.000000000',
       '2022-01-03T18:00:00.000000000', '2022-01-03T21:00:00.000000000',
       '2022-01-04T00:00:00.000000000', '2022-01-04T03:00:00.000000000',
       '2022-01-04T06:00:00.000000000', '2022-01-04T09:00:00.000000000',
       '2022-01-04T12:00:00.000000000', '2022-01-04T15:00:00.000000000',
       '2022-01-04T18:00:00.000000000', '2022-01-04T21:00:00.000000000',
       '2022-01-05T00:00:00.000000000', '2022-01-05T03:00:00.000000000',
       '2022-01-05T06:00:00.000000000', '2022-01-05T09:00:00.000000000',
       '2022-01-05T12:00:00.000000000', '2022-01-05T15:00:00.000000000',
       '2022-01-05T18:00:00.000000000', '2022-01-05T21:00:00.000000000',
       '2022-01-06T00:00:00.000000000', '2022-01-06T03:00:00.000000000',
       '2022-01-06T06:00:00.000000000', '2022-01-06T09:00:00.000000000',
       '2022-01-06T12:00:00.000000000', '2022-01-06T15:00:00.000000000',
       '2022-01-06T18:00:00.000000000', '2022-01-06T21:00:00.000000000',
       '2022-01-07T00:00:00.000000000', '2022-01-07T03:00:00.000000000',
       '2022-01-07T06:00:00.000000000', '2022-01-07T09:00:00.000000000',
       '2022-01-07T12:00:00.000000000', '2022-01-07T15:00:00.000000000',
       '2022-01-07T18:00:00.000000000', '2022-01-07T21:00:00.000000000',
       '2022-01-08T00:00:00.000000000', '2022-01-08T03:00:00.000000000',
       '2022-01-08T06:00:00.000000000', '2022-01-08T09:00:00.000000000',
       '2022-01-08T12:00:00.000000000', '2022-01-08T15:00:00.000000000',
       '2022-01-08T18:00:00.000000000', '2022-01-08T21:00:00.000000000',
       '2022-01-09T00:00:00.000000000', '2022-01-09T03:00:00.000000000',
       '2022-01-09T06:00:00.000000000', '2022-01-09T09:00:00.000000000',
       '2022-01-09T12:00:00.000000000', '2022-01-09T15:00:00.000000000',
       '2022-01-09T18:00:00.000000000', '2022-01-09T21:00:00.000000000',
       '2022-01-10T00:00:00.000000000', '2022-01-10T03:00:00.000000000',
       '2022-01-10T06:00:00.000000000', '2022-01-10T09:00:00.000000000',
       '2022-01-10T12:00:00.000000000', '2022-01-10T15:00:00.000000000',
       '2022-01-10T18:00:00.000000000', '2022-01-10T21:00:00.000000000',
       '2022-01-11T00:00:00.000000000', '2022-01-11T03:00:00.000000000',
       '2022-01-11T06:00:00.000000000', '2022-01-11T09:00:00.000000000',
       '2022-01-11T12:00:00.000000000', '2022-01-11T15:00:00.000000000',
       '2022-01-11T18:00:00.000000000', '2022-01-11T21:00:00.000000000',
       '2022-01-12T00:00:00.000000000', '2022-01-12T03:00:00.000000000',
       '2022-01-12T06:00:00.000000000', '2022-01-12T09:00:00.000000000',
       '2022-01-12T12:00:00.000000000', '2022-01-12T15:00:00.000000000',
       '2022-01-12T18:00:00.000000000', '2022-01-12T21:00:00.000000000',
       '2022-01-13T00:00:00.000000000', '2022-01-13T03:00:00.000000000',
       '2022-01-13T06:00:00.000000000', '2022-01-13T09:00:00.000000000',
       '2022-01-13T12:00:00.000000000', '2022-01-13T15:00:00.000000000',
       '2022-01-13T18:00:00.000000000', '2022-01-13T21:00:00.000000000',
       '2022-01-14T00:00:00.000000000', '2022-01-14T03:00:00.000000000',
       '2022-01-14T06:00:00.000000000', '2022-01-14T09:00:00.000000000',
       '2022-01-14T12:00:00.000000000', '2022-01-14T15:00:00.000000000',
       '2022-01-14T18:00:00.000000000', '2022-01-14T21:00:00.000000000',
       '2022-01-15T00:00:00.000000000', '2022-01-15T03:00:00.000000000',
       '2022-01-15T06:00:00.000000000', '2022-01-15T09:00:00.000000000',
       '2022-01-15T12:00:00.000000000', '2022-01-15T15:00:00.000000000',
       '2022-01-15T18:00:00.000000000', '2022-01-15T21:00:00.000000000',
       '2022-01-16T00:00:00.000000000', '2022-01-16T03:00:00.000000000',
       '2022-01-16T06:00:00.000000000', '2022-01-16T09:00:00.000000000',
       '2022-01-16T12:00:00.000000000', '2022-01-16T15:00:00.000000000',
       '2022-01-16T18:00:00.000000000', '2022-01-16T21:00:00.000000000',
       '2022-01-17T00:00:00.000000000'], dtype='datetime64[ns]')
Attributes:
    grads_dim:      t
    grads_mapping:  linear
    grads_size:     129
    grads_min:      00z01jan2022
    grads_step:     3hr
    long_name:      time
    minimum:        00z01jan2022
    maximum:        00z17jan2022
    resolution:     0.125, 'lev': <xarray.IndexVariable 'lev' (lev: 41)>
array([1.00e+03, 9.75e+02, 9.50e+02, 9.25e+02, 9.00e+02, 8.50e+02, 8.00e+02,
       7.50e+02, 7.00e+02, 6.50e+02, 6.00e+02, 5.50e+02, 5.00e+02, 4.50e+02,
       4.00e+02, 3.50e+02, 3.00e+02, 2.50e+02, 2.00e+02, 1.50e+02, 1.00e+02,
       7.00e+01, 5.00e+01, 4.00e+01, 3.00e+01, 2.00e+01, 1.50e+01, 1.00e+01,
       7.00e+00, 5.00e+00, 3.00e+00, 2.00e+00, 1.00e+00, 7.00e-01, 4.00e-01,
       2.00e-01, 1.00e-01, 7.00e-02, 4.00e-02, 2.00e-02, 1.00e-02])
Attributes:
    grads_dim:      z
    grads_mapping:  levels
    units:          millibar
    long_name:      altitude
    minimum:        1000.0
    maximum:        0.01
    resolution:     24.99975, 'lat': <xarray.IndexVariable 'lat' (lat: 721)>
array([-90.  , -89.75, -89.5 , ...,  89.5 ,  89.75,  90.  ])
Attributes:
    grads_dim:      y
    grads_mapping:  linear
    grads_size:     721
    units:          degrees_north
    long_name:      latitude
    minimum:        -90.0
    maximum:        90.0
    resolution:     0.25, 'lon': <xarray.IndexVariable 'lon' (lon: 1440)>
array([0.0000e+00, 2.5000e-01, 5.0000e-01, ..., 3.5925e+02, 3.5950e+02,
       3.5975e+02])
Attributes:
    grads_dim:      x
    grads_mapping:  linear
    grads_size:     1440
    units:          degrees_east
    long_name:      longitude
    minimum:        0.0
    maximum:        359.75
    resolution:     0.25, 'absvprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) absolute vorticity [1/s] , 'no4lftxsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface best (4 layer) lifted index [k] , 'acpcpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface convective precipitation [kg/m^2] , 'albdosfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface albedo [%] , 'apcpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface total precipitation [kg/m^2] , 'capesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface convective available potential energy [j/kg] , 'cape180_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 180-0 mb above ground convective available potential energ..., 'cape90_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 90-0 mb above ground convective available potential energy..., 'cape255_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 255-0 mb above ground convective available potential energ..., 'cfrzravesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical freezing rain [-] , 'cfrzrsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical freezing rain [-] , 'cicepavesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical ice pellets [-] , 'cicepsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical ice pellets [-] , 'cinsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface convective inhibition [j/kg] , 'cin180_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 180-0 mb above ground convective inhibition [j/kg] , 'cin90_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 90-0 mb above ground convective inhibition [j/kg] , 'cin255_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 255-0 mb above ground convective inhibition [j/kg] , 'clwmrprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) cloud mixing r..., 'clwmrhy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level cloud mixing ratio [kg/kg] , 'cnwatsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface plant canopy surface water [kg/m^2] , 'cpofpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface percent frozen precipitation [%] , 'cpratavesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface convective precipitation rate [kg/m^2/s] , 'cpratsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface convective precipitation rate [kg/m^2/s] , 'crainavesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical rain [-] , 'crainsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical rain [-] , 'csnowavesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical snow [-] , 'csnowsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface categorical snow [-] , 'cwatclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere (considered as a single layer) cloud wat..., 'cworkclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere (considered as a single layer) cloud wor..., 'dlwrfsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface downward long-wave rad. flux [w/m^2] , 'dpt2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground dew point temperature [k] , 'dswrfsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface downward short-wave radiation flux [w/m^2] , 'dzdtprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) vertical velocity (geo..., 'fldcpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface field capacity [fraction] , 'fricvsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface frictional velocity [m/s] , 'gfluxsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface ground heat flux [w/m^2] , 'grleprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) graupel [kg/kg] , 'grlehy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level graupel [kg/kg] , 'gustsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface wind speed (gust) [m/s] , 'hcdcavehcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** high cloud layer high cloud cover [%] , 'hcdchcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** high cloud layer high cloud cover [%] , 'hgtsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface geopotential height [gpm] , 'hgtprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) geopotential height [g..., 'hgt2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface geopotential height [gpm] , 'hgtneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface geopotential height [gpm] , 'hgttop0c': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** highest tropospheric freezing level geopotential height [g..., 'hgtceil': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** cloud ceiling geopotential height [gpm] , 'hgt0c': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0c isotherm geopotential height [gpm] , 'hgtmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind geopotential height [gpm] , 'hgttrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause geopotential height [gpm] , 'hindexsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface haines index [numeric] , 'hlcy3000_0m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 3000-0 m above ground storm relative helicity [m^2/s^2] , 'hpblsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface planetary boundary layer height [m] , 'icahtmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind icao standard atmosphere reference height [m] , 'icahttrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause icao standard atmosphere reference height [m] , 'icecsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface ice cover [proportion] , 'iceg_10m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 10 m above mean sea level ice growth rate [m/s] , 'icetksfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface ice thickness [m] , 'icetmpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface ice temperature [k] , 'icmrprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) ice water mixi..., 'icmrhy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level ice water mixing ratio [kg/kg] , 'landsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface land cover (0=sea, 1=land) [proportion] , 'lcdcavelcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** low cloud layer low cloud cover [%] , 'lcdclcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** low cloud layer low cloud cover [%] , 'lftxsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface surface lifted index [k] , 'lhtflsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface latent heat net flux [w/m^2] , 'mcdcavemcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** middle cloud layer medium cloud cover [%] , 'mcdcmcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** middle cloud layer medium cloud cover [%] , 'msletmsl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** mean sea level mslp (eta model reduction) [pa] , 'o3mrprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) ozone mixing ratio [kg..., 'pevprsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface potential evaporation rate [w/m^2] , 'plpl255_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 255-0 mb above ground pressure of level from which parcel ..., 'potsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level potential temperature [k] , 'prateavesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface precipitation rate [kg/m^2/s] , 'pratesfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface precipitation rate [kg/m^2/s] , 'preslclb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** low cloud bottom level pressure [pa] , 'preslclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** low cloud top level pressure [pa] , 'presmclb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** middle cloud bottom level pressure [pa] , 'presmclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** middle cloud top level pressure [pa] , 'preshclb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** high cloud bottom level pressure [pa] , 'preshclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** high cloud top level pressure [pa] , 'pressfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface pressure [pa] , 'pres80m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 80 m above ground pressure [pa] , 'pres2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface pressure [pa] , 'presneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface pressure [pa] , 'prescclb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** convective cloud bottom level pressure [pa] , 'prescclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** convective cloud top level pressure [pa] , 'presmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind pressure [pa] , 'prestrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause pressure [pa] , 'prmslmsl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** mean sea level pressure reduced to msl [pa] , 'pwatclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere (considered as a single layer) precipita..., 'refcclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere composite reflectivity [db] , 'refd4000m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 4000 m above ground reflectivity [db] , 'refd1000m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1000 m above ground reflectivity [db] , 'refdhy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level reflectivity [db] , 'refdhy2': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 hybrid level reflectivity [db] , 'rhprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) relative humidity [%] , 'rh2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground relative humidity [%] , 'rhsg330_1000': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.33-1 sigma layer relative humidity [%] , 'rhsg440_1000': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.44-1 sigma layer relative humidity [%] , 'rhsg720_940': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.72-0.94 sigma layer relative humidity [%] , 'rhsg440_720': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.44-0.72 sigma layer relative humidity [%] , 'rhsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level relative humidity [%] , 'rh30_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30-0 mb above ground relative humidity [%] , 'rhclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere (considered as a single layer) relative ..., 'rhtop0c': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** highest tropospheric freezing level relative humidity [%] , 'rh0c': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0c isotherm relative humidity [%] , 'rwmrprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) rain mixing ra..., 'rwmrhy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level rain mixing ratio [kg/kg] , 'sfcrsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface surface roughness [m] , 'shtflsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface sensible heat net flux [w/m^2] , 'snmrprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) snow mixing ra..., 'snmrhy1': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1 hybrid level snow mixing ratio [kg/kg] , 'snodsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface snow depth [m] , 'soill0_10cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0-0.1 m below ground liquid volumetric soil moisture (non ..., 'soill10_40cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.1-0.4 m below ground liquid volumetric soil moisture (no..., 'soill40_100cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.4-1 m below ground liquid volumetric soil moisture (non ..., 'soill100_200cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1-2 m below ground liquid volumetric soil moisture (non fr..., 'soilw0_10cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0-0.1 m below ground volumetric soil moisture content [fra..., 'soilw10_40cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.1-0.4 m below ground volumetric soil moisture content [f..., 'soilw40_100cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.4-1 m below ground volumetric soil moisture content [fra..., 'soilw100_200cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1-2 m below ground volumetric soil moisture content [fract..., 'sotypsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface soil type [-] , 'spfhprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) specific humidity [kg/..., 'spfh2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground specific humidity [kg/kg] , 'spfh80m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 80 m above ground specific humidity [kg/kg] , 'spfh30_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30-0 mb above ground specific humidity [kg/kg] , 'sunsdsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface sunshine duration [s] , 'tcdcaveclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere total cloud cover [%] , 'tcdcblcll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** boundary layer cloud layer total cloud cover [%] , 'tcdcclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere total cloud cover [%] , 'tcdcprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 250 200 150 100 50) total cloud co..., 'tcdcccll': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** convective cloud layer total cloud cover [%] , 'tmax2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground maximum temperature [k] , 'tmin2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground minimum temperature [k] , 'tmplclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** low cloud top level temperature [k] , 'tmpmclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** middle cloud top level temperature [k] , 'tmphclt': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** high cloud top level temperature [k] , 'tmpsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface temperature [k] , 'tmpprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) temperature [k] , 'tmp_1829m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1829 m above mean sea level temperature [k] , 'tmp_2743m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2743 m above mean sea level temperature [k] , 'tmp_3658m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 3658 m above mean sea level temperature [k] , 'tmp2m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground temperature [k] , 'tmp80m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 80 m above ground temperature [k] , 'tmp100m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 100 m above ground temperature [k] , 'tmpsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level temperature [k] , 'tmp30_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30-0 mb above ground temperature [k] , 'tmp2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface temperature [k] , 'tmpneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface temperature [k] , 'tmpmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind temperature [k] , 'tmptrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause temperature [k] , 'tozneclm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** entire atmosphere (considered as a single layer) total ozo..., 'tsoil0_10cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0-0.1 m below ground soil temperature validation to deprec..., 'tsoil10_40cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.1-0.4 m below ground soil temperature validation to depr..., 'tsoil40_100cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.4-1 m below ground soil temperature validation to deprec..., 'tsoil100_200cm': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1-2 m below ground soil temperature validation to deprecat..., 'ugwdsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface zonal flux of gravity wave stress [n/m^2] , 'uflxsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface momentum flux, u-component [n/m^2] , 'ugrdprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) u-component of wind [m..., 'ugrd_1829m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1829 m above mean sea level u-component of wind [m/s] , 'ugrd_2743m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2743 m above mean sea level u-component of wind [m/s] , 'ugrd_3658m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 3658 m above mean sea level u-component of wind [m/s] , 'ugrd10m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 10 m above ground u-component of wind [m/s] , 'ugrd20m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 20 m above ground u-component of wind [m/s] , 'ugrd30m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30 m above ground u-component of wind [m/s] , 'ugrd40m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 40 m above ground u-component of wind [m/s] , 'ugrd50m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 50 m above ground u-component of wind [m/s] , 'ugrd80m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 80 m above ground u-component of wind [m/s] , 'ugrd100m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 100 m above ground u-component of wind [m/s] , 'ugrdsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level u-component of wind [m/s] , 'ugrd30_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30-0 mb above ground u-component of wind [m/s] , 'ugrd2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface u-component of wind [m/s] , 'ugrdneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface u-component of wind [m/s] , 'ugrdpbl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** planetary boundary layer u-component of wind [m/s] , 'ugrdmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind u-component of wind [m/s] , 'ugrdtrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause u-component of wind [m/s] , 'ulwrfsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface upward long-wave rad. flux [w/m^2] , 'ulwrftoa': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** top of atmosphere upward long-wave rad. flux [w/m^2] , 'ustm6000_0m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 6000-0 m above ground u-component storm motion [m/s] , 'uswrfsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface upward short-wave radiation flux [w/m^2] , 'uswrftoa': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** top of atmosphere upward short-wave radiation flux [w/m^2] , 'vgwdsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface meridional flux of gravity wave stress [n/m^2] , 'vegsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface vegetation [%] , 'vflxsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface momentum flux, v-component [n/m^2] , 'vgrdprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) v-component of wind [m..., 'vgrd_1829m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 1829 m above mean sea level v-component of wind [m/s] , 'vgrd_2743m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2743 m above mean sea level v-component of wind [m/s] , 'vgrd_3658m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 3658 m above mean sea level v-component of wind [m/s] , 'vgrd10m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 10 m above ground v-component of wind [m/s] , 'vgrd20m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 20 m above ground v-component of wind [m/s] , 'vgrd30m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30 m above ground v-component of wind [m/s] , 'vgrd40m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 40 m above ground v-component of wind [m/s] , 'vgrd50m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 50 m above ground v-component of wind [m/s] , 'vgrd80m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 80 m above ground v-component of wind [m/s] , 'vgrd100m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 100 m above ground v-component of wind [m/s] , 'vgrdsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level v-component of wind [m/s] , 'vgrd30_0mb': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 30-0 mb above ground v-component of wind [m/s] , 'vgrd2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface v-component of wind [m/s] , 'vgrdneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface v-component of wind [m/s] , 'vgrdpbl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** planetary boundary layer v-component of wind [m/s] , 'vgrdmwl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** max wind v-component of wind [m/s] , 'vgrdtrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause v-component of wind [m/s] , 'vissfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface visibility [m] , 'vratepbl': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** planetary boundary layer ventilation rate [m^2/s] , 'vstm6000_0m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 6000-0 m above ground v-component storm motion [m/s] , 'vvelprs': <xarray.Variable (time: 129, lev: 41, lat: 721, lon: 1440)>
[5491251360 values with dtype=float32]
Attributes:
    long_name:  ** (1000 975 950 925 900.. 10 7 4 2 1) vertical velocity (pre..., 'vvelsig995': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 0.995 sigma level vertical velocity (pressure) [pa/s] , 'vwsh2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=2e-06 (km^2/kg/s) surface vertical speed shear [1/s] , 'vwshneg2pv': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** pv=-2e-06 (km^2/kg/s) surface vertical speed shear [1/s] , 'vwshtrop': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** tropopause vertical speed shear [1/s] , 'watrsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface water runoff [kg/m^2] , 'weasdsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface water equivalent of accumulated snow depth [kg/m^2] , 'wiltsfc': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** surface wilting point [fraction] , 'var00212m': <xarray.Variable (time: 129, lat: 721, lon: 1440)>
[133932960 values with dtype=float32]
Attributes:
    long_name:  ** 2 m above ground desc [unit] })
```