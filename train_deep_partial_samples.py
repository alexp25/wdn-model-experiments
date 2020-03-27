import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import deep_learning, loader
import tensorflow as tf
from keras import backend as K
import time
import os
from shutil import copyfile, copy2
import yaml

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]
# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]

if config["run_clean"]:
    loader.clean(root_crt_model_folder)

n_reps = 10
use_saved_model = False
append_timestamp = True
save_best_model = True

if n_reps > 1:
    use_saved_model = False
    append_timestamp = True
    save_best_model = True
else:
    save_best_model = False

# bookmarks = [bookmarks[-1]]
from_bookmark_index = len(bookmarks) - 1

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y = loader.load_dataset(data_file)

    x_train = x
    y_train = y
    x_eval = x
    y_eval = y

    sizex = np.shape(x_train)

    for bookmark_index in range(len(bookmarks)):
        if bookmark_index < from_bookmark_index:
            continue
        x_train = x[0:bookmarks[bookmark_index], :]
        y_train = y[0:bookmarks[bookmark_index], :]
        x_eval = x[0:bookmarks[len(bookmarks)-1], :]
        y_eval = y[0:bookmarks[len(bookmarks)-1], :]

        print(y)
        # binarize the outputs
        y = loader.binarize(y)
        print(y)

        top_acc = 0
        top_model_filename = None

        # run multiple evaluations (each training may return different results in terms of accuracy)
        for i in range(n_reps):
            print("evaluating model rep: " + str(i) + "/" + str(n_reps))

            # session = K.get_session()
            model_file = root_crt_model_folder + "/" + filename
            model_file_raw = model_file

            model_file_raw += "_" + str(bookmark_index)
            if append_timestamp:
                app = "_" + str(time.time())
                model_file_raw += app

            model_file = model_file_raw
            model_file += ".h5"

            # create tensorflow graph session
            graph = tf.Graph()
            with tf.Session(graph=graph):
                if use_saved_model:
                    model = deep_learning.dl_load_model(model_file)
                else:
                    model = deep_learning.create_model(x_train, y_train, config["activation_fn"], config["loss_fn"])
                    deep_learning.dl_save_model(model, model_file)
                    acc_train = deep_learning.eval_model_acc(
                        model, x_train, y_train)
                    acc_eval = deep_learning.eval_model_acc(
                        model, x_eval, y_eval)
                    deep_learning.write_file(model_file, "accuracy: " +
                                             str(acc_eval * 100) + "\r\ntrain: " + str(acc_train * 100))
                    # deep_learning.eval_write_info(
                    #     model, x_eval, y_eval, model_file)

                acc = deep_learning.eval_model(model, x_eval, y_eval, sizex[1])
                if acc > top_acc:
                    top_acc = acc
                    top_model_filename = model_file_raw

                if i == n_reps - 1:
                    if save_best_model:
                        copy2(top_model_filename + ".h5",
                              top_model_filename + "_top.h5")
                        copy2(top_model_filename + ".h5.txt",
                              top_model_filename + "_top.h5.txt")

            # K.clear_session()
