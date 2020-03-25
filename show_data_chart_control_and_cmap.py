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

from modules.preprocessing import get_samples_skip, get_samples_nbins


def get_intersection_matrix(elements: List[CMapMatrixElement], rows, cols):
    intersection_matrix = np.zeros((rows, cols))
    print(rows, cols)
    print("check elems")
    for e in elements:
        print(e.i, e.j)
        intersection_matrix[e.j][e.i] = e.val

    return intersection_matrix


def plot_intersection_matrix_ax(elements, index, nrows, ncols, scale, fig, ax, annotate, cmap, xlabels, ylabels, xlabel, ylabel):
    graph.plot_matrix_cmap_plain_ax(
        elements, nrows, ncols, "", xlabel, ylabel,  xlabels, ylabels, scale, fig, ax, annotate, cmap)

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
            new_row.append(x[i, ord])
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

    data = data[120:1700]
    # data = data[100:]

    # data1, header1 = order_data(data, header, range(14, 19))
    data1, header1 = order_data(data, header, [33])
    data2, header2 = order_data(data, header, [10, 20])
    data3, header3 = order_data(data, header, [35])

    print(np.shape(data1))
    print(np.shape(data2))

    data1_orig = data1

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
        e.i = 0
        elements_all.append(e)

    elements_models = []

    print("data1:")
    n_bins = 20

    data1 = [int(d[0]) for d in data1]
    # print(data1)
    data1 = get_samples_nbins(data1, n_bins)

    print(data1)
    print(len(data1))

    print("data2:")
    # print(elements_combined)
    print(len(elements_combined))
    # quit()

    model_max = data1[0]

    # e = CMapMatrixElement()
    # e.i = 0
    # e.j = len(data1)
    # e.val = 0
    # elements_models.append(e)

    for i, d in enumerate(data1):
        e = CMapMatrixElement()
        e.i = 0
        e.j = i
        e.val = d
        if d > model_max:
            model_max = d
        elements_models.append(e)

    # for e in elements_models:
    #     e.val = model_max - e.val

    # elements_models 
    elements_models[0].val = 0
    elements_combined[0].val = 0

    # for e in elements_combined:
    #     if e.val != 0:
    #         e.val = 1 / e.val

    # e = CMapMatrixElement()
    # e.i = 0
    # e.j = len(data1)
    # e.val = 0
    # elements_models.append(e)

    # xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
    # xlabels.append("  ")
    # xlabels.append("p ")

    nrows = 3
    ncols = 8

    nrows = 1
    ncols = n_bins

    ncols += 1

    fig, ax = graph.get_n_ax(4, (10, 9), [5, 5, 1, 1])
    # fig, ax = graph.get_n_ax(3, None)

    
    qs = [0]

    lastq = data1_orig[0]
    # print(lastq)
    for i, d in enumerate(data1_orig):
        if d != lastq:
            lastq = d
            qs.append(i)


    graph.plot_timeseries_ax(tss2[0], "", "", "flow [L/h]", fig, ax[0], qs)
    graph.plot_timeseries_ax(tss3[0], "", "", "pump [%]", fig, ax[1], qs)

    # tab10
    plot_intersection_matrix_ax(
        elements_models, 0, nrows, ncols, None, fig, ax[2], False, "viridis_r", ["model "], [(" ") for i in range(n_bins)], "", "")

    plot_intersection_matrix_ax(
        elements_combined, 1, nrows, ncols, (0, 1), fig, ax[3], False, "Blues", ["acc % "], [(" ") for i in range(n_bins)], "samples [x0.1s]", "")


    graph.show_fig(fig)

    graph.save_figure(fig, "./figs/control_integration" + filename)
