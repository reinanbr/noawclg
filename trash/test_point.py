import noawclg.main as main
from noawclg.main import get_noaa_data as gnd
from trash.plot import plot_data_from_place as pdp
import pandas as pd
import matplotlib.pyplot as plt


date_base = '23/09/2023'

main.set_date(date_base)
data_noaa = gnd()#,url_data='https://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs20220108/gfs_1p00_00z')

place = 'juazeiro BA'

jua_pet = pdp(place=place,data=data_noaa)

jua_pet.path_file='plot_wind100m.png'

jua_pet.key_noaa='tmp80m'
jua_pet.title='Temperatura do Ar\nPetrolina-PE/Juazeiro-BA'
jua_pet.ylabel='°C'
jua_pet.xlabel='Set/Out de 2023'

def fmt_data(data): return data-273
jua_pet.fmt_data =  fmt_data

plt = jua_pet.render()

plt.axvline(pd.Timestamp('2023-09-23'),lw=2,linestyle='--',c='r')

plt.axvline(pd.Timestamp('2023-10-01'),lw=2,linestyle='--',c='r')


plt.annotate('Início da Onda de Calor\nno País', (jua_pet.index[16],jua_pet.m_temp[27]), xytext=(20, 17), 
            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))


plt.annotate('Fím* da Onda de Calor\nno País', (jua_pet.index[80],jua_pet.m_temp[37]), xytext=(20, 36), 
            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

plt.savefig('test.png')
#plt.show()

