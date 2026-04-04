import json

dir = 'data_info/JUAZEIRO_BA.json'
dir_new = 'data_info/juazeiro_ba.json'

file_write = open(dir_new,'w')


file_read = open(dir,'r')
data = json.load(file_read)
list_date = list(data["date"].values())
list_hour = list(data["hour"].values())


new_data = {}
i = 0
for dt in list_date:
    key_dt = f'{list_date[i]}_{list_hour[i]}'
    new_data[key_dt] = {}
    for key in data:
        new_data[key_dt][key] = list(data[key].values())[i]
    i = i + 1



json.dump(new_data,file_write,indent=4)

