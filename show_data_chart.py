import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import loader, graph
from modules.graph import Timeseries
import time
import os
import yaml
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]
 
# filenames = ["exp_39"]

root_data_folder += "/selected"
# filenames = ["exp_345", "exp_350", "exp_352"]
# filenames = ["exp_combined"]
filenames = ["exp_345"]

n_reps = 1
use_saved_model = False
append_timestamp = True
save_best_model = True

if n_reps > 1:
    use_saved_model = False
    append_timestamp = True
    save_best_model = True
else:
    save_best_model = False


def remove_outliers(data):

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    for j in range(cols):   
        for i in range(rows):
            if i > 1:
                if data[i][j] > 1200:
                    data[i][j] = data[i-1][j]

    return data

def reorder(x, order):
  

    x_ord = []

    for (i, ord) in enumerate(order):
        x_ord.append(x[ord])

    return np.array(x_ord)

def reorder2d(x, order):
    sdata = np.shape(x)
    rows = sdata[0]
    cols = sdata[1]

    x_ord = []
    for i in range(rows):
        new_row = []
        for (j, ord) in enumerate(order):
            new_row.append(x[i,ord])
        x_ord.append(new_row)
        
    return np.array(x_ord)

def create_timeseries(data, header):
    tss: List[Timeseries] = []
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']
    
    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

    for j in range(cols):
        ts: Timeseries = Timeseries()
        ts.label = header[j]
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0
        
        for i in range(rows):
            ts.x.append(i)
            ts.y.append(data[i][j])

        tss.append(ts)
        ts = None
        
    return tss

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, z, xheader, yheader, zheader = loader.load_dataset_3(data_file)

    # tss = create_timeseries(x, xheader)

    # TODO: sort by chan number 0 - 10
    # TODO: show as subplot 

    print(xheader)
    print(yheader)

    print(len(xheader))

    order = [0,1,3,4,5,6,7,8,9,10,2]

    xheader = reorder(xheader, order)

    x = reorder2d(x, order)

    print(x)

    # xheader[2], xheader[10] = xheader[10], xheader[2]
    # x[:, 2], x[:, 10] = x[:, 10], x[:, 2].copy()

    print("sorted")
    print(xheader)
    print(yheader)


    # x = x[467:1035, :]
    # y = y[467:1035, :]
    # z = z[]

    print(x)

    print(np.shape(x))
    print(np.shape(y))

    # tss = create_timeseries(np.concatenate((x,y), axis=1), np.concatenate((xheader,yheader)))

    tss = create_timeseries(y, yheader)

    x = remove_outliers(x)

    tss2 = create_timeseries(x, xheader)
    
    # print(json.dumps(acc, indent=2))

    # fig, _ = graph.plot_timeseries_multi(tss, "valve sequence", "samples [x0.1s]", "position [%]", False)

    fig, _ = graph.plot_timeseries_multi_sub2([tss, tss2], ["valve sequence", "sensor output"], "samples [x0.1s]", ["position [%]", "flow [L/h]"])

    graph.save_figure(fig, "./figs/valve_sequence_" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

  
    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()

    