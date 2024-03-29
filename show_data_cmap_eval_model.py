from modules import graph

from modules.graph import Timeseries, CMapMatrixElement
import numpy as np

from os import listdir
from os.path import isfile, join
import json
from typing import List

from modules.preprocessing import Preprocessing
from modules import generator

from modules import deep_learning, loader
import tensorflow as tf
from keras import backend as K

import yaml
import pickle

import modules.mat_aux as maux

# import copy

with open("config.yml", "r") as f:
    config = yaml.load(f)


root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]

# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]
model_filenames = filenames

# root_data_folder += "/random1"
# root_crt_model_folder = "./data/models/deep_rnn_random"
# filenames = ["exp_179"]
# model_filenames = ["exp_179"]

# set this as in saved models folder
n_reps = 5

results_vect_train = []
results_vect_test = []

use_rnn = True

prep = Preprocessing()

output_filename = "eval_deep_1_"
if use_rnn:
    output_filename = "eval_deep_2_rnn_"

# output_filename = "eval_deep_3_rnn_random_"
# output_filename = "eval_deep_5_rnn_random_"

if config["one_hot_encoding"]:
    binv = generator.generate_binary(config["n_valves"])
    print("binv:")
    print(binv)
    binv = prep.adapt_input(binv)
    print("adapted:")
    print(binv)
    # print("to list")
    # print(prep.str_to_list(binv))
    prep.create_encoder(binv)

# quit()


use_random_exp = True
use_matching_random_model = True
from_file = True

eval_rowskip = False

# run twice, with False/True to extract the data into .dat files
use_post_rowskip = True


if not use_random_exp:
    use_matching_random_model = False

if config["load_from_container"]:
    if use_rnn:
        root_crt_model_folder = config["root_model_container"] + \
            "/deep_rnn"
        if use_matching_random_model:
            root_crt_model_folder += "_random"
    else:
        root_crt_model_folder = config["root_model_container"] + "/deep"

elements: List[CMapMatrixElement] = []
elements_prediction: List[CMapMatrixElement] = []

rowsdict = {}
colsdict = {}

if use_random_exp:
    # input_file = "./data/random1/raw_buffer.csv"
    input_file = "./data/random1/exp_179.csv"
    # input_file = "./data/control/2/exp_217.csv"
    if use_matching_random_model:
        model_file = root_crt_model_folder + "/" + "exp_179_1_top.h5"
        # model_file = root_crt_model_folder + "/" + "exp_217_2_top.h5"
    else:
        model_file = root_crt_model_folder + "/" + "exp_39_4_top.h5"
else:
    input_file = "./data/exp_39.csv"
    model_file = root_crt_model_folder + "/" + "exp_39_5_top.h5"

nvalves = config["n_valves"]


nrowskip = 0

# X1, y1 = loader.load_dataset_raw_buffer(input_file)
X1, y1, _, _ = loader.load_dataset(input_file)

# X1 = X1[120:1700]
# y1 = y1[120:1700]

# binarize the outputs
y1 = loader.binarize(y1)

s = np.shape(X1)
print(s)

nrows = s[0]
ncols = s[1]


n_bins = 20
rowskip = int(nrows/n_bins)

if use_post_rowskip:
    rowskip = 1

post_rowskip = int(nrows/n_bins)

s2 = np.shape(y1)
yrows = s2[0]
ycols = s2[1]
print(s2)

X = []
y = []

nrows2 = 0
rowindex = 0

for r in range(nrows):
    elements_buffer = []

    for c in range(ycols):
        e = CMapMatrixElement()
        e.i = c
        e.j = r
        e.val = int(y1[r][c])
        elements_buffer.append(e)

    comp_eval = not eval_rowskip

    nrowskip += 1

    if nrowskip >= rowskip:
        nrowskip = 0
        comp_eval = True
        for e in elements_buffer:
            e.j = rowindex
            elements.append(e)
        rowindex += 1

    if comp_eval:
        y.append([])
        X.append([])
        for c in range(ycols):
            y[nrows2].append(int(y1[r][c]))
        for c in range(ncols):
            X[nrows2].append(int(X1[r][c]))

        nrows2 += 1

y_orig = y

if config["one_hot_encoding"]:
    y = prep.encode(prep.adapt_input(y))

X = np.array(X)

sizex = np.shape(X)
sizey = np.shape(y)

print("refactored: ")
print(sizex)
print(sizey)


# create tensorflow graph session
tfgraph = tf.Graph()
with tf.Session(graph=tfgraph):
    model = deep_learning.dl_load_model(model_file)
    acc = deep_learning.eval_model(
        model, X, y, sizex[1], use_rnn)

    # make probability predictions with the model
    predictions = deep_learning.predict_model_RNN(model, X)
    sizep = np.shape(predictions)

    print("prediction shape: ")
    print(sizep)

    if config["one_hot_encoding"]:
        predictions = np.reshape(predictions, (sizep[0], sizep[2]))
        sizep = np.shape(predictions)
        print("reshape predictions: ")
        print(sizep)

    predictions = deep_learning.binarize_predictions_max(predictions)

    predictions_str = prep.adapt_input(predictions)
    # print(predictions_str[0:10])
    predictions_orig = predictions

    if config["one_hot_encoding"]:
        predictions = prep.decode(predictions)
        predictions = prep.str_to_list(predictions)

print("accuracy: ")
print(acc)

match = []
nomatch = []

print(predictions[0:10])

sizep = np.shape(predictions)
print(sizep)

nrows = sizep[0]
ncols = sizep[1]

match_count = 0

print(predictions[0])
print(predictions[0][0])
print(y_orig[0])
print(y_orig[0][0])

multiple_ones_count = 0
nrowskip = 0
rowindex = 0

# get the matching predictions for the original encoding
for i in range(nrows):
    m = True
    elements_buffer: List[CMapMatrixElement] = []
    n_ones = np.sum(predictions[i])
    if n_ones > 1:
        multiple_ones_count += 1
        # print("multiple ones: ", predictions[i])

    for j in range(ncols):
        e = CMapMatrixElement()
        e.i = j
        e.j = i
        e.val = predictions[i][j]
        elements_buffer.append(e)
        if predictions[i][j] != y_orig[i][j]:
            m = False
            break

    nrowskip += 1

    if nrowskip >= rowskip:
        nrowskip = 0
        for e in elements_buffer:
            e1 = CMapMatrixElement()
            e1.i = e.i
            e1.j = rowindex
            e1.val = e.val
            elements_prediction.append(e1)
        rowindex += 1
    if m:
        match_count += 1
    else:
        nomatch.append("".join([str(e) for e in predictions[i]]) +
                       " > expected > " + "".join([str(e) for e in y_orig[i]]))

    match.append(m)

print(str(match_count) + "/" + str(len(predictions)))
print("Real accuracy: ", match_count/len(predictions)*100)

print(nomatch[0:10])

print("multiple ones: ", multiple_ones_count)

print(len(elements))
print(len(elements_prediction))

if not use_post_rowskip:
    nrows = n_bins

xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
# ylabels = [(str(int(i * rowskip / 100))) for i in range(n_bins)]
ylabels = [(" ") for i in range(n_bins)]

intersection_matrix = maux.get_intersection_matrix_elems(
    elements, nrows, nvalves)
intersection_matrix_prediction = maux.get_intersection_matrix_elems(
    elements_prediction, nrows, nvalves)

# quit()

savefig = True


def plot_intersection_matrix(elements: List[CMapMatrixElement], index, save):
    # fig = graph.plot_matrix_cmap_plain(
    #     elements, nvalves, nrows, "Valve Sequence", "sample x" + str(rowskip), "valves",  xlabels, ylabels)
    fig = graph.plot_matrix_cmap_plain(
        elements, nvalves, nrows, "", "", "valves",  xlabels, ylabels, (16,7))
    if save:
        graph.save_figure(fig, "./figs/valve_sequence_" + str(index))


intersection_matrix_buffer = [
    [0 for i in range(nvalves)] for j in range(post_rowskip)]


eqrows = 0
acc_combined = 0

if use_post_rowskip:
    avg_rows_list = []
    nrowskip = 0

    for row in range(nrows):
        row1 = [e.val if e is not None else 0 for e in intersection_matrix[row]]
        row2 = [
            e.val if e is not None else 0 for e in intersection_matrix_prediction[row]]

        if maux.check_equal_rows(row1, row2):
            eqrows += 1
            for col in range(nvalves):
                intersection_matrix_buffer[nrowskip][col] += 1

        nrowskip += 1

        if nrowskip >= post_rowskip:
            nrowskip_crt = nrowskip
            # print(nrowskip_crt)
            nrowskip = 0
            avg_rows: List[int] = []
            for col in range(nvalves):
                avg_rows.append(0)

                for r in range(nrowskip_crt):
                    val = intersection_matrix_buffer[r][col]
                    avg_rows[col] += val
                    intersection_matrix_buffer[r][col] = 0

                avg_rows[col] /= nrowskip_crt

            avg_rows_list.append(avg_rows)
            print(avg_rows)
            acc_combined += avg_rows[0]

    nrows = len(avg_rows_list)
    ncols = len(avg_rows_list[0])

    elements_combined: List[CMapMatrixElement] = maux.get_elems_from_matrix(
        avg_rows_list, nrows, ncols)

    print(eqrows)
    print(len(avg_rows_list))
    print(len(elements_combined))
    print(acc_combined/len(avg_rows_list))


def plot_intersection_matrix2(elements, index, nrows, ncols, save, scale):
    fig = graph.plot_matrix_cmap_plain(
        elements, nrows, ncols, "", "", "",  xlabels, ylabels, scale, None)
    if save:
        graph.save_figure(fig, "./figs/valve_sequence_" + str(index))

if not use_post_rowskip:
    with open("elem1.dat", "wb") as f:
        pickle.dump(elements, f)
    with open("elem2.dat", "wb") as f:
        pickle.dump(elements_prediction, f)

    plot_intersection_matrix(elements, 1, savefig)
    plot_intersection_matrix(elements_prediction, 2, savefig)
else:
    # elements_combined = elements_combined[0:n_bins]

    elements_combined = [e for e in elements_combined if e.i == 0]
    # for (i, e) in enumerate(elements_combined):
    #     e.i = 0
    #     e.j = i
    #     elements.append(e)

    # print(elements_combined)

    vals = [e.val for e in elements_combined]
    print(vals)
    print("avg: ", np.mean(vals))

    n_bins = len(elements_combined)

    with open("elem.dat", "wb") as f:
        pickle.dump(elements_combined, f)
