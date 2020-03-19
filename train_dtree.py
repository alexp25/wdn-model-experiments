import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import loader, model_loader, graph
from modules.graph import Barseries
import time
import os
from shutil import copyfile, copy2
import yaml
import json
from modules.preprocessing import Preprocessing
from modules import generator
from typing import List

# TODO: https://stackoverflow.com/questions/21556623/regression-with-multi-dimensional-targets

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]
# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]

n_reps = 5
use_saved_model = False
append_timestamp = False
save_best_model = True

if use_saved_model:
    n_reps = 1

acc_train_vect = {}
acc_test_vect = {}

prep = Preprocessing()

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

if config["run_clean"] and not use_saved_model:
    loader.clean(root_crt_model_folder)

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)

    acc_train_vect[filename] = {
        "data": [],
        "aux": [],
        "files": [],
        "acc": [],
        "avg": 0
    }
    acc_test_vect[filename] = {
        "data": [],
        "aux": [],
        "files": [],
        "acc": [],
        "avg": 0
    }

    # print(y)
    # binarize the outputs
    y = loader.binarize(y)

    if config["one_hot_encoding"]:
        y = prep.encode(prep.adapt_input(y))
        
    # print(y)

    top_acc = 0
    top_model_filename = None

    # session = K.get_session()

    # classifiers.create_decision_tree(x, y[:,0], 20)
    sizey = np.shape(y)

    for rep in range(n_reps):

        acc_train_vect[filename]["data"].append([])
        acc_train_vect[filename]["aux"].append([])
        acc_test_vect[filename]["data"].append([])
        acc_test_vect[filename]["aux"].append([])
        acc_train_vect[filename]["files"].append([])
        acc_test_vect[filename]["files"].append([])

        for (i, s) in enumerate(range(sizey[1])):

            model_file = root_crt_model_folder + "/" + filename
            model_file_raw = model_file
            model_file_raw += "_" + str(rep+1) + "_" + str(i+1)

            if append_timestamp:
                app = "_" + str(time.time())
                model_file_raw += app

            model_file = model_file_raw + ".skl"

            # X = x[:, i]
            X = x
            yi = y[:, i]

            # print(np.shape(x))
            # print(np.shape(y))

            n_train_percent = config["train_percent"]

            x_train, y_train = classifiers.split_dataset_train(
                X, yi, n_train_percent)
            x_test, y_test = classifiers.split_dataset_test(
                X, yi, n_train_percent)

            dt = 0

            if not use_saved_model:
                tstart = time.time()
                model = classifiers.create_decision_tree()
                model, acc = classifiers.train_decision_tree(
                    model, x_train, y_train)
                dt = time.time() - tstart
            else:
                model = model_loader.load_sklearn_model(model_file)

            model, acc, diff, total = classifiers.predict_decision_tree(
                model, x_train, y_train, False)

            acc_train_vect[filename]["data"][rep].append(diff)
            acc_train_vect[filename]["aux"][rep].append(total)
            acc_train_vect[filename]["files"][rep].append(model_file)
            acc_train_vect[filename]["acc"].append(acc)

            model, acc, diff, total = classifiers.predict_decision_tree(
                model, x_test, y_test, False)

            model_loader.save_sklearn_model(model, model_file)
            acc_test_vect[filename]["data"][rep].append(diff)
            acc_test_vect[filename]["aux"][rep].append(total)
            acc_test_vect[filename]["files"][rep].append(model_file)
            acc_test_vect[filename]["acc"].append(acc)


def set_avg(dc):
    # for each input file (experiment)
    for dc1 in dc:
        acc_vect = []
        for (rep, data) in enumerate(dc[dc1]["data"]):
            ds = np.array(dc[dc1]["data"][rep])
            dsc = np.array(dc[dc1]["aux"][rep])
            # the accuracy of the experiment is the average accuracy of each decision tree
            acc = np.sum(ds) / np.sum(dsc) * 100

            # acc = dc[dc1]["acc"][rep]
            acc_vect.append(acc)

        acc_vect = np.array(acc_vect)
        dc[dc1]["avg"] = np.average(acc_vect)
        dc[dc1]["top"] = np.max(acc_vect)
        dc[dc1]["top_file"] = dc[dc1]["files"][0][np.argmax(acc_vect)]


set_avg(acc_train_vect)
set_avg(acc_test_vect)

print("\ntrain vect")
print(json.dumps(acc_train_vect))
print("\ntest vect")
print(json.dumps(acc_test_vect))


def create_barseries(accs, keys):
    tss: List[Barseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    ck = 0
    for (i, acc) in enumerate(accs):
        ts: Barseries = Barseries()
        ts.label = keys[i]
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        ts.data = []
        for (j, key) in enumerate(acc):
            ts.data.append(acc[key]["avg"])

        print(ts.data)
        tss.append(ts)
        ts = None
    return tss


def extract_csv(vect):
    csvoutput = ""
    for e in vect:
        # print(e)
        csvoutput += e + "," + \
            str(vect[e]["avg"]) + "," + str(vect[e]["top"]) + \
            "," + str(vect[e]["top_file"]) + "\n"

    return csvoutput


csvoutput = extract_csv(acc_train_vect)

with open("./data/output/eval_dtree_1_train.csv", "w") as f:
    f.write(csvoutput)

csvoutput = extract_csv(acc_test_vect)

with open("./data/output/eval_dtree_1_test.csv", "w") as f:
    f.write(csvoutput)

# print(csvoutput)

tss = create_barseries([acc_train_vect, acc_test_vect], ["train", "test"])

fig = graph.plot_barchart_multi(tss, "model", "accuracy", "Average accuracy (decision tree)", [
                                "1-N-80%", "1-N-1-80%", "1-N-1-50%", "GRAY-80%"], False)

graph.save_figure(fig, "./figs/mean_accuracy_dtree_1")
