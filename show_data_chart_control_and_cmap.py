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


import pickle
from modules import graph

from modules.graph import Timeseries, CMapMatrixElement
import numpy as np

from os import listdir
from os.path import isfile, join
import json
from typing import List

import yaml


def get_intersection_matrix(elements: List[CMapMatrixElement], rows, cols):
    intersection_matrix = np.zeros((rows, cols))
    print(rows, cols)
    print("check elems")
    for e in elements:
        print(e.i, e.j)
        intersection_matrix[e.j][e.i] = e.val

    return intersection_matrix


def plot_intersection_matrix_ax(elements, index, nrows, ncols, save, scale, fig, ax):
    graph.plot_matrix_cmap_plain_ax(
        elements, nrows, ncols, "", "", "",  xlabels, ylabels, scale, fig, ax)
    
def plot_intersection_matrix(elements, index, nrows, ncols, save, scale):
    fig = graph.plot_matrix_cmap_plain(
        elements, nrows, ncols, "", "", "",  xlabels, ylabels, scale)
    if save:
        graph.save_figure(fig, "./figs/valve_sequence_random_prediction")


nvalves = 6






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


    with open("elem.dat", "rb") as f:
        elements_combined = pickle.load(f)

    with open("elem1.dat", "rb") as f:
        elements = pickle.load(f)

    with open("elem2.dat", "rb") as f:
        elements_prediction = pickle.load(f)


    n_bins = len(elements_combined)
    rowskip = 1

    imatrix = get_intersection_matrix(elements_combined, n_bins, 1)
    print(imatrix)

    elements_all = []

    for e in elements:
        elements_all.append(e)

    for e in elements_combined:
        e.i = 7
        elements_all.append(e)


    xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
    xlabels.append("  ")
    xlabels.append("p ")
    # ylabels = [("x" + str(i+1)) for i in range(n_bins)]
    ylabels = [(" ") for i in range(n_bins)]


    fig, ax = graph.get_n_ax(3, (8, 16))
    # fig, ax = graph.get_n_ax(3, None)

    plot_intersection_matrix_ax(
        elements_all, 3, 8, n_bins, True, (0, 1), fig, ax[0])

    graph.plot_timeseries_ax(tss2[0], "title", "xlabel", "ylabel", fig, ax[1])
    graph.plot_timeseries_ax(tss3[0], "title", "xlabel", "ylabel", fig, ax[2])

    graph.show_fig(fig)

    graph.save_figure(fig, "./figs/control_integration" + filename)


    