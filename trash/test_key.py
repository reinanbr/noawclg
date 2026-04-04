from trash import get_noaa_data as  gnd

keys=(gnd().get_noaa_keys())


for key in keys.keys():
    print(f'{key}: {keys[key]}')
