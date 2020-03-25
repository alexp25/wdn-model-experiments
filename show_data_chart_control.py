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
filenames = config["filenames"]
bookmarks = config["bookmarks"]

root_data_folder = "./data/control/2"
filenames = ["exp_217"]


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
    colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']
    
    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    # colors = cm.rainbow(np.linspace(0, 1, cols))
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

def order_data(data, header, order):

    header1 = reorder(header, order)
    data1 = reorder2d(data, order)

    return data1, header1

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    data, header = loader.load_dataset_full_with_header(data_file)
  
    print(header)
    print(len(header))
    for i, h in enumerate(header):
        print(i, h)

    # order = [2,3,5,6,7,8,9,10,11,12,4]
    # order = [10, 20]

    data = data[100:]

    # data1, header1 = order_data(data, header, range(14, 19))
    data1, header1 = order_data(data, header, [33])
    data2, header2 = order_data(data, header, [10, 20])
    data3, header3 = order_data(data, header, [35])
    
    print(np.shape(data1))
    print(np.shape(data2))
   
    # tss = create_timeseries(np.concatenate((x,y), axis=1), np.concatenate((xheader,yheader)))

    tss1 = create_timeseries(data1, header1)

    # x = remove_outliers(x)

    tss2 = create_timeseries(data2, header2)
    tss3 = create_timeseries(data3, header3)
    
    # print(json.dumps(acc, indent=2))

    # fig, _ = graph.plot_timeseries_multi(tss, "valve sequence", "samples [x0.1s]", "position [%]", False)

    fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3], ["valve sequence", "sensor output", "pump output"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]"])

    graph.save_figure(fig, "./figs/control" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

  
    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()

    