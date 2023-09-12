import json
import os
import shutil

path_to_sailvos = './sailvos/'
path_to_sailvoscut = './sailvos_cut/'
if not os.path.exists(path_to_sailvoscut):
    os.mkdir(path_to_sailvoscut)
# Read data from file:
data = json.load( open( "/sailvos_cut_structure.json" ) )
keys = data.keys()

for cut_id in sorted(keys):
    sailvos_origin_video = cut_id.split('__')[0]
    frames = data[cut_id]
    for frame in frames:
        path_src = path_to_sailvos + '/' + sailvos_origin_video + '/'+ 'images/' + frame
        if not os.path.exists(path_to_sailvoscut + '/' + cut_id):
            os.mkdir(path_to_sailvoscut + '/' + cut_id)
        if not os.path.exists(path_to_sailvoscut + '/' + cut_id + '/images/'):
            os.mkdir(path_to_sailvoscut + '/' + cut_id + '/images/')
        path_dts = path_to_sailvoscut + '/' + cut_id + '/' + 'images/'+ frame
        shutil.copy(path_src, path_dts)


