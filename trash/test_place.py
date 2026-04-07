import noawclg.main as main
import pandas as pda
from datetime import datetime


def dict_place_data_base_climate(place,date_base):
    
    main.set_date(date_base)
    dn = main.get_noaa_data()
    pd = dn.get_data_from_place(place)

    humidity = pd['rh2m']
    temp = pd['tmp2m']-272
    temp_air = pd['tmp100m'] - 272
    temp_jet_air = pd['tmpmwl'] - 272
    rain =pd['pwatclm']
    cloud = pd['tcdcclm']
    v_j = pd['ugrd10m']
    v_k = pd['vgrd10m']
    v = pd['gustsfc']
    time = pd['time']
    pressure = pd['pressfc']
    dt = []
    hour = []
    times = []
    for tm_ in time:

        tm = tm_.to_numpy()
        
        dt_ = datetime.strptime(str(tm)[:19], "%Y-%m-%dT%H:%M:%S")
        dt.append(dt_.strftime("%d/%m/%Y"))
        hour.append(dt_.strftime("%H:%M"))
        times.append(dt_.timestamp())
    
    dict_res = {
        'time':times,
        'date':dt,
        'hour':hour,
        'humidity':humidity,
        'temperature':temp,
        'air_temperature':temp_air,
        'jet_air_temperature':temp_jet_air,
        'rain_percent':rain,
        'pressure':pressure,
        'cloud':cloud,
        'v_k':v_k,
        'v_j':v_j,
        'v':v
    }
    
    return pda.DataFrame(dict_res)

dt_now = datetime.now().strftime('%d/%m/%Y')
list_city = ['JUAZEIRO BA']
for city in list_city:
    print(f'making data for {city}')
    data_jua = dict_place_data_base_climate(city,dt_now)
    file_city = city.replace(' ','_')
    data_jua.to_excel(f'data_info/{file_city}.xlsx')
    data_jua.to_json(f'data_info/{file_city}.json')
