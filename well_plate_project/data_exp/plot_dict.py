#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:48:32 2020

@author: enzo
"""

from well_plate_project.config import data_dir
path_data =  data_dir / 'raw' / 'exp_v2_crp' /'luce_nat' /'10 (bkp2)'
json_file_path = path_data / '20201118_090416_0.84_dict.json'
assert json_file_path.is_file()

import json
with open(str(json_file_path)) as json_file:
    data = json.load(json_file)
    
lists = sorted(data.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples


[x in data.values() for y in data.values()]

[x.values() for x in data.values()]


df=pd.DataFrame({'abbr':list(currency_dict.keys()),
                 'curr':list(currency_dict.values())})


all = [ele for ww in ['a', 'b'] for ele in data[ww] ]


a=[list(x.keys()) for x in data.values()]


list(set(a))
