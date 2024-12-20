
<h1 align='center'>NOAWClg</h1>
<p align='center'>

<br/>
<a href="https://github.com/perseu912"><img title="Autor" src="https://img.shields.io/badge/Autor-reinan_br-blue.svg?style=for-the-badge&logo=github"></a>
<!-- <br/>
<a href='http://dgp.cnpq.br/dgp/espelhogrupo/0180330616769073'><img src='https://shields.io/badge/cnpq-grupo_de_fisica_computacional_ifsertao--pe-blueviolet?logo=appveyor&style=for-the-badge'></a> -->

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

<br>

#### Problem with netcdf?


try:

```sh
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libnetcdf-dev


export HDF5_DIR=/usr/local/hdf5
export HDF5_DIR=/usr/include/hdf5

pip install netcdf4
pip install -U xarray #24/11/24
```

or

```sh
sudo apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev

export HDF5_DIR=/usr/local/hdf5
export HDF5_DIR=/usr/include/hdf5

pip install netcdf4
```
<hr>

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
       '2022-01-01T12:00:00.000000000', 
...
```

### keys
you can see the all keys in <a href='https://github.com/reinanbr/noawclg/blob/main/key.log'>it page.</a> 
```py
>>> from noawclg import get_noaa_data as  gnd

>>> gnd().get_noaa_keys()


{'time': 'time', 
'lev': 'altitude', 
'lat': 'latitude', 
'lon': 'longitude', 
'absvprs': '** (1000 975 950 925 900.. 10 7 4 2 1) absolute vorticity [1/s] ',
 'no4lftxsfc': '** surface best (4 layer) lifted index [k] ', 
 'acpcpsfc': '** surface convective precipitation [kg/m^2] ', 
 'albdosfc': '** surface albedo [%] ',
  'apcpsfc': '** surface total precipitation [kg/m^2] ', 
  'capesfc': '** surface convective available potential energy [j/kg] ', 
...
```

### example plot wind
```py
import noawclg.main as main
from noawclg.main import get_noaa_data as gnd
from noawclg.plot import plot_data_from_place as pdp
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#reinan voltou, porrrrraaaaaaaaa

date_base = '12/01/2023'

main.set_date(date_base)
data_noaa = gnd()#,url_data='https://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs20220108/gfs_1p00_00z')

place = 'juazeiro BA'

jua_pet = pdp(place=place,data=data_noaa)

jua_pet.path_file='plot_wind100m.png'

jua_pet.key_noaa='tmp80m'
jua_pet.title='Temperatura do Ar\nPetrolina-PE/Juazeiro-BA'
jua_pet.ylabel='°C'
jua_pet.xlabel='Janeiro de 2023'

def fmt_data(data): return data-273
jua_pet.fmt_data =  fmt_data

jua_pet.render()
#plt.show()



```
result:
<img src='https://raw.githubusercontent.com/reinanbr/noawclg/main/plot_wind100m.png'/>



<!-- the truest Reinan 13/01/23 01:58 :siga a rotina criada -->
<!-- N se molde por ninguem e pra ninguem -->
<!-- se precisar, abndone todos os complexos-->
<!-- fique com qualquer uma, transe com quiem quiser transar ctg 10:15 14/01/23-->
<!-- e demonstre isso em suas redes sociais, para vim mais-->



<img src="https://reysofts.com.br/engine/libs/save_table_access_libs.php?lib_name=noawclg">
