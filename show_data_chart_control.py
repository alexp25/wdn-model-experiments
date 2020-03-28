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


def get_flow_metrics(data2):
    err = [abs(e[0] - e[1]) for e in data2]
    mean = np.mean([e[0] for e in data2])
    stdev = np.std(err)
    # print(mean, stdev)
    return mean, stdev


def get_control_metrics(data3):
    d = [e[0] for e in data3]
    mean = np.mean(d)
    stdev = np.std(d)
    # print(mean, stdev)
    return mean, stdev


def split_data_bookmarks(data):
    bookmarks = []
    b1 = data[0]
    i1 = 0
    for i, d in enumerate(data):
        if d != b1 and i > 0:
            b1 = data[i-1]
            i2 = i-1
            if i1 != i2:
                bookmarks.append([i1, i2])
            i1 = i
    return bookmarks

def run_multiple_bookmarks(data, bm, fn):
    res = []
    for b in bm:
        d = data[b[0]:b[1]]
        res.append(fn(d))
    return res

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

    data = data[200:]

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

    gmean_flow, gstdev_flow = get_flow_metrics(data2)
    print(gmean_flow, gstdev_flow)
    gmean_pump, gstdev_pump = get_control_metrics(data3)
    print(gmean_pump, gstdev_pump)
    # print(data2)
    bm = split_data_bookmarks(data1)
    print(bm)
    # quit()

    header_aux = ["crt", "mean"]
    res = run_multiple_bookmarks(data2, bm, get_flow_metrics)
    res1 = [[e[0], gmean_flow] for e in res]
    flow_mean_bm = [e[0] for e in res]
    flow_min_bm = np.min(flow_mean_bm)
    flow_max_bm = np.max(flow_mean_bm)
    gmean_flow_bm = np.mean(flow_mean_bm)

    print(flow_min_bm)
    print(flow_max_bm)
    print(gmean_flow_bm)
    print(gmean_flow)

    print(abs(flow_max_bm - gmean_flow_bm)/gmean_flow_bm * 100)
    print(abs(flow_min_bm - gmean_flow_bm)/gmean_flow_bm * 100)

    print(abs(flow_max_bm - gmean_flow)/gmean_flow * 100)
    print(abs(flow_min_bm - gmean_flow)/gmean_flow * 100)


    gstdev_pump_bm = np.mean(gstdev_pump)
    print(gstdev_pump_bm)

    # quit()


    tss4 = create_timeseries(res1, header_aux)
    res2 = [[e[1], gstdev_flow] for e in res]
    tss5 = create_timeseries(res2, header_aux)

    res = run_multiple_bookmarks(data3, bm, get_control_metrics)
    res1 = [[e[0], gmean_pump] for e in res]
    tss6 = create_timeseries(res1, header_aux)
    res2 = [[e[1], gstdev_pump] for e in res]
    tss7 = create_timeseries(res2, header_aux)

    print("shape:")
    print(np.shape(data2))
    print(np.shape(res1))


    # head = ["", "", "", ""]
    # axhead = ["yk (avg)", "yk (stdev)", "uk (avg)", "uk (stdev)"]
    # fig, _ = graph.plot_timeseries_multi_sub2([tss4, tss5, tss6, tss7], head, "samples [x0.1s]", axhead)


    # quit()
    fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3], [
                                              "valve sequence", "sensor output", "pump output"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]"])

    # fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3, tss4, tss5], [
    #                                           "valve sequence", "sensor output", "pump output", "mean", "stdev"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]", "mean [L/h]", "stdev"])


    graph.save_figure(fig, "./figs/control" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()
