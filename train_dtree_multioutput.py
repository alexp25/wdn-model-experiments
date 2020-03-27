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
import copy
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

acc_train_vect = {}
acc_test_vect = {}

use_randomforest = False

if use_randomforest:
    output_filename = "dtree_2_multioutput"
else:
    output_filename = "dtree_1"

prep = Preprocessing()

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

if config["run_clean"] and not use_saved_model:
    loader.clean(root_crt_model_folder)


def init_vect(vect):
    for key in vect["data"]:
        # print(key)
        vect["data"][key].append(None)

    vect["count"].append(None)
    vect["files"].append(None)


def update_vect(vect, index, acc, count, dt, fsize, file1):
    vect["count"][index] = count
    vect["files"][index] = file1
    vect["data"]["acc"][index] = acc
    vect["data"]["dt"][index] = dt
    vect["data"]["fsize"][index] = fsize


dmodel1 = {
    "acc": [],
    "dt": [],
    "fsize": []
}

dmodel = {
    "data": copy.deepcopy(dmodel1),
    "count": [],
    "files": [],
    "avg": copy.deepcopy(dmodel1),
    "top": copy.deepcopy(dmodel1)
}

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)

    acc_train_vect[filename] = copy.deepcopy(dmodel)
    acc_test_vect[filename] = copy.deepcopy(dmodel)

    # print(y)
    # binarize the outputs
    y = loader.binarize(y)
    if config["one_hot_encoding"]:
        # use integer encoding
        y = prep.encode(prep.adapt_input(y))
        y = prep.decode_int_onehot(y)

        # print(y)
        # quit()
        # y = prep.encode(prep.adapt_input(y))

    top_acc = 0
    top_model_filename = None

    # session = K.get_session()

    # classifiers.create_decision_tree(x, y[:,0], 20)
    sizey = np.shape(y)

    for rep in range(n_reps):

        init_vect(acc_train_vect[filename])
        init_vect(acc_test_vect[filename])

        model_file = root_crt_model_folder + "/" + filename
        model_file_raw = model_file
        model_file_raw += "_" + str(rep+1)

        if use_randomforest:
            model_file_raw += "_multi"

        if append_timestamp:
            app = "_" + str(time.time())
            model_file_raw += app

        model_file = model_file_raw + ".skl"

        # X = x[:, i]
        X = x

        # print(np.shape(x))
        # print(np.shape(y))

        n_train_percent = config["train_percent"]

        x_train, y_train = classifiers.split_dataset_train(
            X, y, n_train_percent)
        x_test, y_test = classifiers.split_dataset_test(
            X, y, n_train_percent)

        dt = 0

        if not use_saved_model:
            tstart = time.time()
            model = classifiers.create_multi_output_classifier(
                use_randomforest)
            model, acc = classifiers.train_decision_tree(
                model, x_train, y_train)
            dt = time.time() - tstart
        else:
            model = model_loader.load_sklearn_model(model_file)

        model, acc, diff, total = classifiers.predict_decision_tree(
            model, x_train, y_train, False)

        # update training results

        model_loader.save_sklearn_model(model, model_file)

        fsize = os.stat(model_file).st_size

        update_vect(acc_train_vect[filename], rep,
                    diff, total, dt, 0, model_file)

        tstart = time.time()
        model, acc, diff, total = classifiers.predict_decision_tree(
            model, x_test, y_test, False)
        dt_eval = time.time() - tstart

        # update test results
        update_vect(acc_test_vect[filename], rep, diff,
                    total, dt_eval, fsize, model_file)

        if acc > top_acc:
            top_acc = acc
            top_model_filename = model_file_raw

        if rep == n_reps - 1:
            if save_best_model:
                copy2(top_model_filename + ".skl",
                      top_model_filename + "_top.skl")


# def set_avg(dc):
#     # for each input file (experiment)
#     for dc1 in dc:
#         acc_vect = []
#         for (rep, data) in enumerate(dc[dc1]["data"]):
#             ds = dc[dc1]["data"][rep]
#             dsc = dc[dc1]["aux"][rep]
#             # the accuracy of the experiment is the average accuracy of each decision tree
#             acc = ds / dsc * 100
#             acc_vect.append(acc)

#         acc_vect = np.array(acc_vect)
#         dc[dc1]["avg"] = np.average(acc_vect)
#         dc[dc1]["top"] = np.max(acc_vect)
#         dc[dc1]["top_file"] = dc[dc1]["files"][0][np.argmax(acc_vect)]

def set_avg(dc):
    # for each input file (experiment)
    for dc1 in dc:
        acc_vect = copy.deepcopy(dmodel1)
        for rep in range(len(dc[dc1]["count"])):
            count = dc[dc1]["count"][rep]
            for key in dmodel1:
                # the accuracy of the experiment is the average accuracy of each decision tree
                ds = dc[dc1]["data"][key][rep]

                if key == "acc":
                    avg_ds = ds / count
                    avg_ds *= 100
                else:
                    avg_ds = float(ds)

                acc_vect[key].append(avg_ds)

        for key in dmodel1:
            acc_v1 = np.array(acc_vect[key])
            print(key, acc_v1)
            dc[dc1]["avg"][key] = np.average(acc_v1)
            dc[dc1]["top"][key] = np.max(acc_v1)

            if key == "acc":
                dc[dc1]["top_file"] = dc[dc1]["files"][np.argmax(acc_v1)]


set_avg(acc_train_vect)
set_avg(acc_test_vect)

print("\ntrain vect")
print(json.dumps(acc_train_vect, indent=2))
print("\ntest vect")
print(json.dumps(acc_test_vect, indent=2))


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
            ts.data.append(acc[key]["avg"]["acc"])

        print(ts.data)
        tss.append(ts)
        ts = None
    return tss


def extract_csv(vect):
    csvoutput = ""
    for e in vect:
        # print(e)
        csvoutput += e + "," + \
            str(vect[e]["avg"]["acc"]) + "," + str(vect[e]["top"]["acc"]) + \
            "," + str(vect[e]["avg"]["dt"]) + "," + str(vect[e]["avg"]
                                                        ["fsize"]) + "," + str(vect[e]["top_file"]) + "\n"

    return csvoutput


csvoutput = extract_csv(acc_train_vect)

with open("./data/output/eval_" + output_filename + "_train.csv", "w") as f:
    f.write(csvoutput)

csvoutput = extract_csv(acc_test_vect)

with open("./data/output/eval_" + output_filename + "_test.csv", "w") as f:
    f.write(csvoutput)

# print(csvoutput)

tss = create_barseries([acc_train_vect, acc_test_vect], ["train", "test"])

fig = graph.plot_barchart_multi(tss, "model", "accuracy", "Average accuracy (decision tree)", [
                                "1-N-80%", "1-N-1-80%", "1-N-1-50%", "GRAY-80%"], [70, 120])

graph.save_figure(fig, "./figs/mean_accuracy_" + output_filename)
