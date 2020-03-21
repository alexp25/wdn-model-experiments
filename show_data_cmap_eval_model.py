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

if config["load_from_container"]:
    if use_rnn:
        root_crt_model_folder = config["root_model_container"] + \
            "/deep_rnn"
    else:
        root_crt_model_folder = config["root_model_container"] + "/deep"

elements: List[CMapMatrixElement] = []
elements_prediction: List[CMapMatrixElement] = []

rowsdict = {}
colsdict = {}

input_file = "./data/random1/raw_buffer.csv"
input_file = "./data/exp_39.csv"
# model_file = root_crt_model_folder + "/" + "exp_39_4_top.h5"
model_file = root_crt_model_folder + "/" + "exp_39_5_top.h5"

nvalves = config["n_valves"]

rowskip = 500
nrowskip = 0

eval_rowskip = False

# X1, y1 = loader.load_dataset_raw_buffer(input_file)
X1, y1, _, _ = loader.load_dataset(input_file)

# binarize the outputs
y1 = loader.binarize(y1)

s = np.shape(X1)
print(s)

nrows = s[0]
ncols = s[1]

s2 = np.shape(y1)
yrows = s2[0]
ycols = s2[1]
print(s2)

X = []
y = []

nrows2 = 0

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
            elements.append(e)

    if comp_eval:
        y.append([])
        X.append([])
        for c in range(ycols):
            y[nrows2].append(int(y1[r][c]))
        for c in range(ncols):
            X[nrows2].append(int(X1[r][c]))

        nrows2 += 1

y_orig = y

# print("refactored: ")
# s = np.shape(y_orig)
# print(s[0], s[1])
# s = np.shape(X)
# print(s[0], s[1])

# quit()

if config["one_hot_encoding"]:
    y = prep.encode(prep.adapt_input(y))

X = np.array(X)

sizex = np.shape(X)
sizey = np.shape(y)

print("refactored: ")
print(sizex)
print(sizey)

# quit()

# y = prep.decode(y)
# y = prep.str_to_list(y)

# print(prep.decode_int_onehot(y))
# quit()


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

    # predictions = deep_learning.binarize_predictions(predictions, 0.2, 0.8)
    predictions = deep_learning.binarize_predictions(predictions, 0.4, 0.6)
    predictions_str = prep.adapt_input(predictions)
    # print(predictions_str[0:10])
    predictions_orig = predictions

    if config["one_hot_encoding"]:
        # print("predictions binary encoded to int")
        # print(predictions[0:10])
        # print(prep.decode_int_onehot(predictions_orig))
        # decode onehot
        predictions = prep.decode(predictions)
        predictions = prep.str_to_list(predictions)

print("accuracy: ")
print(acc)

# print(predictions)

match = []
nomatch = []

print(predictions[0:10])

# quit()

# use one hot encoded data
# predictions = predictions_orig
# y_orig = y
#

sizep = np.shape(predictions)
print(sizep)

nrows = sizep[0]
ncols = sizep[1]

match_count = 0

print(predictions[0])
print(predictions[0][0])
print(y_orig[0])
print(y_orig[0][0])


# get the matching predictions for the original encoding
for i in range(nrows):
    m = True
    elements_buffer: List[CMapMatrixElement] = []
    for j in range(ncols):
        e = CMapMatrixElement()
        e.i = j
        e.j = i
        e.val = predictions[i][j]
        elements_buffer.append(e)
        if predictions[i][j] != y_orig[i][j]:
            m = False
            break
    for e in elements_buffer:
        elements_prediction.append(e)
    if m:
        match_count += 1
    else:
        nomatch.append("".join([str(e) for e in predictions[i]]) +
                       " > expected > " + "".join([str(e) for e in y_orig[i]]))

    match.append(m)

print(str(match_count) + "/" + str(len(predictions)))
print("Real accuracy: ", match_count/len(predictions)*100)

print(nomatch[0:10])

# quit()
# print(elements)

quit()


xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
ylabels = [(str(int(i * rowskip / 100))) for i in range(nrows)]
# xlabels = []
# ylabels = []


def plot_intersection_matrix(elements: List[CMapMatrixElement], index, save):
    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
    intersection_matrix = np.zeros((nvalves, nrows))

   # print(intersection_matrix)

    for e in elements:
        intersection_matrix[e.i][e.j] = e.val

    print(intersection_matrix)

    fig = graph.plot_matrix_cmap_plain(
        elements, nvalves, nrows, "Valve Sequence", "sample x" + str(rowskip), "valves",  xlabels, ylabels)
    if save:
        graph.save_figure(fig, "./figs/valve_sequence_" + str(index))


savefig = True
plot_intersection_matrix(elements, 0, savefig)
plot_intersection_matrix(elements_prediction, 1, savefig)
